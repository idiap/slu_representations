# -*- coding: UTF-8 -*-
"""
Copyright (c) 2023, Idiap Research Institute (http://www.idiap.ch/)

@author: Esau Villatoro Tello (esau.villatoro@idiap.ch)

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.

NOTICE: Some sections of this code have been adapted from the original
        HERMIT NLU package avaiable at https://arxiv.org/abs/1910.00912
        Authors: Andrea Vanzo, Emanuele Bastianelli, Oliver Lemon
"""

import datetime
import json
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
from prettytable import PrettyTable
from sklearn.metrics import classification_report, f1_score
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

import learning.utils.Constants as Constants
from data.preprocessing import get_y_true_generator, unpadv2
from learning.metrics.sequence_labeling import seqs_classification_report
from learning.SLU_Network import SLU_Hybrid_Network, SLU_Network, SLU_WCN_Network
from learning.utils.bert_xlnet_inputs import prepare_inputs_for_bert_xlnet
from learning.utils.mask_util import prepare_mask

np.random.seed(42)


def get_global_group():
    if dist.is_initialized():
        if not hasattr(get_global_group, "_global_group"):
            get_global_group._global_group = dist.new_group()
        return get_global_group._global_group
    else:
        return None


class Network(object):
    def __init__(
        self,
        network="SLU",
        sentence_lenght=64,
        label_encoders=None,
        num_da_class=10,
        num_intent_class=10,
        num_slot_tags=10,
        dataset_obj=None,
        text_dimensionality=768,
        acoustic_dimensionality=768,
        run_folder="./",
        units=200,
        dropout=0.8,
        use_cuda=False,
        use_wcn=False,
    ):
        self.network = network
        self.sentence_lenght = sentence_lenght
        self.label_encoders = label_encoders
        self.num_da_class = num_da_class
        self.num_intent_class = num_intent_class
        self.num_slot_tags = num_slot_tags
        self.dataset_obj = dataset_obj
        self.text_dim = text_dimensionality
        self.acou_dim = acoustic_dimensionality
        self.checkpoint_path = run_folder
        self.units = units
        self.dropout = dropout
        self.use_cuda = use_cuda
        self.use_wcn = use_wcn
        self.labels = OrderedDict()
        for label in label_encoders:
            self.labels[label] = len(label_encoders[label].classes_)

        self.model = None
        self.evaluation_report = OrderedDict()
        self.tuning_report = OrderedDict()

        # THIS IS FOR WCN Exps , maybe not necessary right now
        # self.model_class, self.tokenizer_class, self.pretrained_weights = (DistilBertModel, DistilBertTokenizerFast, 'distilbert-base-uncased')
        self.model_class, self.tokenizer_class, self.pretrained_weights = (
            BertModel,
            BertTokenizer,
            "bert-base-uncased",
        )
        self.tokenizer = self.tokenizer_class.from_pretrained(
            self.pretrained_weights, do_lower_case=True
        )
        self.pre_trained_model = self.model_class.from_pretrained(
            self.pretrained_weights
        )
        ## ADDING SPECIAL TOKENS TO BERT TOKENIZER [<s>, </s>, <eps>]
        self.tokenizer.add_special_tokens(
            {"bos_token": Constants.BOS_WORD, "eos_token": Constants.EOS_WORD}
        )
        self.tokenizer.add_tokens(["<eps>", "<unk>"], special_tokens=True)
        self.pre_trained_model.resize_token_embeddings(len(self.tokenizer))

        print("Network initialized!")

    def create_network(self, hyper_params):
        if self.network == "SLU":
            self.model = SLU_Network(
                self.sentence_lenght,
                self.label_encoders,
                self.num_da_class,
                self.num_intent_class,
                self.num_slot_tags,
                self.use_cuda,
                dropout=self.dropout,
                units=self.units,
            )
        elif self.network == "SLU_Hybrid":
            self.model = SLU_Hybrid_Network(
                self.sentence_lenght,
                self.label_encoders,
                self.num_da_class,
                self.num_intent_class,
                self.num_slot_tags,
                self.use_cuda,
                self.text_dim,
                self.acou_dim,
                dropout=self.dropout,
                units=self.units,
            )
        elif self.network == "SLU_WCN":
            self.model = SLU_WCN_Network(
                self.sentence_lenght,
                self.label_encoders,
                self.num_da_class,
                self.num_intent_class,
                self.num_slot_tags,
                self.use_cuda,
                dropout=self.dropout,
                units=self.units,
            )

        self.count_parameters(self.model)
        if self.use_cuda:
            self.model.cuda()

    def count_parameters(self, model):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in list(
            filter(lambda p: p[1].requires_grad, model.named_parameters())
        ):
            if not parameter.requires_grad:
                continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params += param
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params

    ##This method works for the Text Only Experiments. It predicts Dialogue acts, intents and slots
    def predict_validation(self, validation_set, batch_size=32):
        val_loss_da = 0
        val_loss_fr = 0
        val_loss_fe = 0
        pred_da = []
        pred_fr = []
        pred_fe = []
        self.model.eval()
        for i, batch in enumerate(
            tqdm(validation_set, desc="\tEvaluating BATCH_DEV_NUMS:")
        ):
            X_val, Y_val, WCN = batch
            X_val_TextEmb = X_val[0]
            X_val_AcousticEmb = X_val[1]
            X_val_masks = X_val[2]
            Y_val_dialogueActs = torch.squeeze(Y_val["dialogue_act"])
            Y_val_frames = torch.squeeze(Y_val["frame"])
            Y_val_frameElements = torch.squeeze(Y_val["frame_element"])
            if self.use_cuda:
                # self.model.cuda()
                X_val_TextEmb = X_val_TextEmb.cuda()
                # X_val_AcousticEmb = X_val_AcousticEmb.cuda()
                X_val_masks = X_val_masks.cuda()
                Y_val_dialogueActs = Y_val_dialogueActs.cuda()
                Y_val_frames = Y_val_frames.cuda()
                Y_val_frameElements = Y_val_frameElements.cuda()
            logits_val_da, logits_val_fr, logits_val_fe = self.model(X_val_TextEmb)
            loss_val_da, loss_val_fr, loss_val_fe = self.model.slu_loss(
                logits_val_da,
                logits_val_fr,
                logits_val_fe,
                Y_val_dialogueActs,
                Y_val_frames,
                Y_val_frameElements,
                mask=X_val_masks,
            )

            pred_val_da, pred_val_fr, pred_val_fe = self.model.predict_slu_tasks(
                logits_val_da, logits_val_fr, logits_val_fe, mask=X_val_masks
            )

            ##updating losses
            val_loss_da += loss_val_da.item()
            val_loss_fr += loss_val_fr.item()
            val_loss_fe += loss_val_fe.item()

            for i in range(len(pred_val_da)):
                pred = []
                for j in range(len(pred_val_da[i])):
                    pred.append(pred_val_da[i][j].item())
                pred_da.append(pred[:])  # [1:-1]

            for i in range(len(pred_val_fr)):
                pred = []
                for j in range(len(pred_val_fr[i])):
                    pred.append(pred_val_fr[i][j].item())
                pred_fr.append(pred[:])  # [1:-1]

            for i in range(len(pred_val_fe)):
                pred = []
                for j in range(len(pred_val_fe[i])):
                    pred.append(pred_val_fe[i][j].item())
                pred_fe.append(pred[:])  # [1:-1]

        data_nums = len(validation_set.dataset)
        ave_loss_da = val_loss_da * batch_size / data_nums
        ave_loss_fr = val_loss_fr * batch_size / data_nums
        ave_loss_fe = val_loss_fe * batch_size / data_nums
        print(
            "\nEvaluation - loss_DA: {:.3f}, loss_INTENT: {:.3f}, loss_SLOTS: {:.3f} \n".format(
                ave_loss_da, ave_loss_fr, ave_loss_fe
            )
        )
        self.model.train()
        return [pred_da, pred_fr, pred_fe]

    ##This method works for the Cross-Modal Experiments. It predicts intents only
    def predict_cross_modal_validation(self, validation_set, batch_size=32):
        val_loss_fr = 0
        pred_fr = []
        true_fr = []
        self.model.eval()
        for i, batch in enumerate(
            tqdm(validation_set, desc="\tEvaluating BATCH_DEV_NUMS:")
        ):
            X_val, Y_val, _ = batch
            X_val_TextEmb = X_val[0]
            X_val_AcousticEmb = X_val[1]
            X_val_masks = X_val[2]
            Y_val_dialogueActs = torch.squeeze(Y_val["dialogue_act"])
            Y_val_frames = torch.squeeze(Y_val["frame"])
            Y_val_frames = self.get_single_label_vector(
                Y_val_frames
            )  # TODO FIXME remove at certain point
            Y_val_frameElements = torch.squeeze(Y_val["frame_element"])
            if self.use_cuda:
                # self.model.cuda()
                X_val_TextEmb = X_val_TextEmb.cuda()
                X_val_AcousticEmb = X_val_AcousticEmb.cuda()
                X_val_masks = X_val_masks.cuda()
                # Y_val_dialogueActs = Y_val_dialogueActs.cuda()
                Y_val_frames = Y_val_frames.cuda()
                # Y_val_frameElements = Y_val_frameElements.cuda()

            logits_val_fr = self.model(X_val_TextEmb, X_val_AcousticEmb)
            loss_val_fr = self.model.slu_single_task_loss(logits_val_fr, Y_val_frames)

            pred_val_fr = self.model.pred_intent(logits_val_fr)
            pred_fr.extend(pred_val_fr.cpu().numpy().tolist())
            true_fr.extend(Y_val_frames.cpu().numpy().tolist())
            ##updating losses
            val_loss_fr += loss_val_fr.item()
        data_nums = len(validation_set.dataset)
        ave_loss_fr = val_loss_fr * batch_size / data_nums
        print("\nEvaluation - loss_INTENT: {:.3f} \n".format(ave_loss_fr))

        self.model.train()
        return true_fr, pred_fr

    ##This works for the Lattice-based experiments, predicts intent detection task using the
    ##WCN module
    def predict_validation_wcn(self, validation_set, batch_size=32):
        val_loss_fr = 0
        pred_da = []
        true_da = []
        self.model.eval()
        for i, batch in enumerate(
            tqdm(validation_set, desc="Evaluating BATCH_DEV_NUMS:")
        ):
            X_val, Y_val, WCN_val = batch

            # Y_val_dialogueActs = torch.squeeze(Y_val['dialogue_act'])
            # Y_val_dialogueActs = self.get_single_label_vector(Y_val_dialogueActs)
            Y_val_frames = torch.squeeze(Y_val["frame"])
            Y_val_frames = self.get_single_label_vector(
                Y_val_frames
            )  # This is necessary to get the single labels
            # Y_val_frameElements = torch.squeeze(Y_val['frame_element'])

            inputs_val = {}
            raw_in_val = WCN_val[0]  ## contains the raw inputs
            batch_pos_val = WCN_val[1]  ## contains the positions vector
            batch_scores_val = WCN_val[2]  # contains the wcn scores
            raw_lens_val = self.sentence_lenght

            if self.use_cuda:
                pretrained_inputs_val = prepare_inputs_for_bert_xlnet(
                    raw_in_val,
                    raw_lens_val,
                    self.tokenizer,
                    batch_pos_val,
                    batch_scores_val,
                    cls_token_at_end=False,
                    cls_token="[CLS]",
                    sep_token="[SEP]",
                    cls_token_segment_id=0,
                    pad_on_left=False,
                    pad_token_segment_id=0,
                    device=torch.device("cuda"),
                )
                # Y_val_dialogueActs = Y_val_dialogueActs.cuda()
                Y_val_frames = Y_val_frames.cuda()
                # Y_val_frameElements = Y_val_frameElements.cuda()

            inputs_val["pretrained_inputs"] = pretrained_inputs_val
            masks_val = prepare_mask(pretrained_inputs_val)

            logits_val_fr = self.model(inputs_val, masks_val)
            loss_val_fr = self.model.slu_single_task_loss(logits_val_fr, Y_val_frames)

            pred_val_da = self.model.pred_dialogue_act(logits_val_fr)
            pred_da.extend(pred_val_da.cpu().numpy().tolist())
            true_da.extend(Y_val_frames.cpu().numpy().tolist())
            ##updating losses
            val_loss_fr += loss_val_fr.item()
        data_nums = len(validation_set.dataset)
        ave_loss_fr = val_loss_fr * batch_size / data_nums
        print("\nEvaluation - loss_INTENT: {:.3f} \n".format(ave_loss_fr))

        self.model.train()
        return true_da, pred_da

    def get_single_label_vector(self, tensor):
        labels = []
        for item in tensor:
            if item.dim != 0:
                labels.append(
                    item[0].item()
                )  ### This is why I was removing the AA-PADDING in the frame labels
            else:
                labels.append(0)
        # print("###NUMBER OF LABELS {}".format(len(set(labels))))
        return torch.tensor(labels)

    """CHECKPOINTING"""

    def checkpoint_model(
        self,
        fname,
        model_state_dict,
        epoch,
        optimizer_state_dict,
        rounds,
        best_epoch,
        best_f1,
        local_f1,
    ):
        rank = os.environ["RANK"]
        if int(rank) == 0:
            print("###SAVING CHECKPOINT")
            print("Saving as:{}".format(os.path.join(self.checkpoint_path, fname)))
            state = {
                "model": model_state_dict,
                "epoch": epoch,
                "optimizer": optimizer_state_dict,
                "rounds_wo_improvement": rounds,
                "best_epoch": best_epoch,
                "total_F1": best_f1,
                "local_F1": local_f1,
            }
            # TODO: add logic to move the latest model to something else
            torch.save(state, os.path.join(self.checkpoint_path, fname))
        # all others should wait. I think it would be better to move it to a separate thread, or make it asynchronous
        dist.barrier()

    """CHECKPOINTING"""

    def load_checkpoint(self, chkpt_name=""):
        chkpt = torch.load(self.get_latest_checkpoint_path(chkpt_name))
        state_dict = chkpt["model"]
        epoch = chkpt["epoch"]
        optimizer = chkpt["optimizer"]
        rounds = chkpt["rounds_wo_improvement"]
        best_epoch = chkpt["best_epoch"]
        bestf1 = chkpt["total_F1"]
        localf1 = chkpt["local_F1"]
        return state_dict, epoch, optimizer, rounds, best_epoch, bestf1, localf1

    """CHECKPOINTING"""

    def get_latest_checkpoint_path(self, chkpt_name=""):
        return os.path.join(self.checkpoint_path, chkpt_name)

    """This verstion of the tune module works for the cross-modal transformers experiment"""

    def tune_cross_modal(
        self, train_set, val_set, epochs, patience, hyper_params, fold=1, n_folds=1
    ):
        print("\n=== Tuning CROSS-MODAL network ===")
        best_f1 = 0
        best_hyper_params = None
        best_epoch = 0
        ## getting validation set labes
        val_y = get_y_true_generator(val_set)
        batch_size = hyper_params[0]["batch_size"]
        for setting_counter, current_hyper_params in enumerate(hyper_params):
            best_hyper_params = current_hyper_params
            print("\nCurrent hyper-params: {}".format(current_hyper_params))
            rounds_without_improvements = 0
            local_best_f1 = 0

            ##checkpointing
            # Creating folder for checkpoints
            self.checkpoint_path = os.path.join(
                "resources", self.checkpoint_path, "checkpoints"
            )
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)

            ##Creating network
            self.create_network(current_hyper_params)
            assert self.model is not None

            ##Checking if checkpoint exists
            last_epoch = 0
            if os.path.exists(
                os.path.join(self.checkpoint_path, "checkpoint_cross_modal.pt")
            ):
                print("\t ##### Restaring from CHECKPOINT")
                (
                    state_dict,
                    last_epoch,
                    optimizer_state_dict,
                    rounds_without_improvements,
                    best_epoch,
                    best_f1,
                    local_best_f1,
                ) = self.load_checkpoint(chkpt_name="checkpoint_cross_modal.pt")
                self.model.load_state_dict(state_dict)

            ##PyTorch
            self.model.train()

            # DDP makes sure the model is the same across all devices
            distributed_model = DistributedDataParallel(
                module=self.model.cuda(),
                process_group=get_global_group(),
                find_unused_parameters=True,
            )
            # optimizer = RMSprop(self.model.parameters(), weight_decay=0.000001)
            optimizer = AdamW(
                distributed_model.parameters(), weight_decay=0.01
            )  # RMSprop(self.model.parameters(), weight_decay=0.000001)
            # if starting from checkpoint, load optimizer values
            if os.path.exists(
                os.path.join(self.checkpoint_path, "checkpoint_cross_modal.pt")
            ):
                optimizer.load_state_dict(optimizer_state_dict)
            scheduler = MultiStepLR(
                optimizer, [20, 50, 100, 150, 200, 250], last_epoch=-1
            )
            ## Running through the epochs
            for epoch in range(last_epoch, epochs):
                print(
                    "Epoch {}/{} (Patience {}/{}) - Setting {}/{} - Fold {}/{}".format(
                        epoch + 1,
                        epochs,
                        rounds_without_improvements,
                        patience,
                        setting_counter + 1,
                        len(hyper_params),
                        fold,
                        n_folds,
                    )
                )
                print(scheduler.get_last_lr())
                step = 0
                # tr_loss_da = 0
                tr_loss_fr = 0
                # tr_loss_fe = 0
                for i, batch in enumerate(tqdm(train_set, desc="Training BATCH_NUMS")):
                    step += 1
                    distributed_model.zero_grad()
                    X, Y, WCN = batch
                    X_TextEmb = X[0]  # contains textual embeddings
                    X_AcousticEmb = X[1]  # contains acoustic embedings
                    X_masks = X[2]  # contains attention masks
                    # Y_dialogueActs = torch.squeeze(Y['dialogue_act'])
                    Y_frames = torch.squeeze(Y["frame"])
                    Y_frames = self.get_single_label_vector(Y_frames)
                    # Y_frameElements = torch.squeeze(Y['frame_element'])

                    assert self.use_cuda
                    if self.use_cuda:
                        X_TextEmb = X_TextEmb.cuda()
                        X_AcousticEmb = X_AcousticEmb.cuda()
                        X_masks = X_masks.cuda()
                        # Y_dialogueActs = Y_dialogueActs.cuda()  #NOT required for CrossMODAL
                        Y_frames = Y_frames.cuda()
                        # Y_frameElements = Y_frameElements.cuda() #NOT required for CrossMODAL

                    ##FORWARD PASS
                    logits_fr = distributed_model(X_TextEmb, X_AcousticEmb)
                    loss_fr = self.model.slu_single_task_loss(logits_fr, Y_frames)
                    tr_loss_fr += loss_fr.item()
                    ## BACKWARD PASS
                    loss = loss_fr
                    loss.backward()
                    ## UPDATING MODEL
                    optimizer.step()

                ave_loss_fr = tr_loss_fr * batch_size / len(train_set.dataset)
                if epoch == 1 or epoch % 10 == 0:
                    print(
                        "\t {} Epoch: {}, TRAINING loss_INTENT: {:.3f} accumulated LOSS: {:.3f}".format(
                            datetime.datetime.now(),
                            epoch + 1,
                            ave_loss_fr,
                            loss.item() * batch_size,
                        )
                    )

                ### Working with the validation set
                true_y, prediction = self.predict_cross_modal_validation(
                    val_set, batch_size=hyper_params[0]["batch_size"]
                )
                f1 = f1_score(true_y, prediction, average="micro")
                total_f1 = f1  # report[current_hyper_params['monitor_m']]['total']

                if total_f1 >= best_f1:
                    best_hyper_params = current_hyper_params
                    best_f1 = total_f1
                    best_epoch = epoch
                if total_f1 >= local_best_f1:
                    local_best_f1 = total_f1
                    rounds_without_improvements = 0
                    print(classification_report(true_y, prediction)),
                    print("")
                else:
                    rounds_without_improvements += 1
                if rounds_without_improvements == patience:
                    break
                ##SAVING CHECK POINT
                if epoch == 2 or epoch % 2 == 0:
                    ##Checkpointing the model
                    self.checkpoint_model(
                        "checkpoint_cross_modal.pt",
                        distributed_model.module.state_dict(),
                        epoch,
                        optimizer.state_dict(),
                        rounds_without_improvements,
                        best_epoch,
                        best_f1,
                        local_best_f1,
                    )
                scheduler.step()
            del local_best_f1
            del rounds_without_improvements
            del self.model
        del val_y
        print("Best F1: {}".format(best_f1))
        print("Best epoch: {}".format(best_epoch + 1))
        print("Best hyper-params: {}".format(best_hyper_params))
        return best_hyper_params, best_epoch + 1

    """This methos is for training usign the large dataset, using the CROSS-MODAL approach, for Intents Only"""

    def train_cross_modal(
        self, train_set, eval_set, epochs, hyper_params, predictions_folder
    ):
        ## This is for the CROSS_MODAL transformer approach
        print("\n=== Training CROSS-MODAL network ===")
        batch_size = hyper_params["batch_size"]
        self.create_network(hyper_params)
        assert self.model is not None
        ##Checking if checkpoint exists
        last_epoch = 0
        if os.path.exists(
            os.path.join(self.checkpoint_path, "checkpoint_MULTMOD_LargeTrain.pt")
        ):
            print(" ## Restaring from CHECKPOINT")
            (
                state_dict,
                last_epoch,
                optimizer_state_dict,
                _,
                _,
                _,
                _,
            ) = self.load_checkpoint(chkpt_name="checkpoint_MULTMOD_LargeTrain.pt")
            self.model.load_state_dict(state_dict)
        self.model.train()
        # DDP makes sure the model is the same across all devices
        distributed_model = DistributedDataParallel(
            module=self.model.cuda(),
            process_group=get_global_group(),
            find_unused_parameters=True,
        )
        optimizer = AdamW(distributed_model.parameters(), weight_decay=0.01)
        # if starting from checkpoint, load optimizer values
        if os.path.exists(
            os.path.join(self.checkpoint_path, "checkpoint_MULTMOD_LargeTrain.pt")
        ):
            optimizer.load_state_dict(optimizer_state_dict)
        scheduler = MultiStepLR(optimizer, [20, 50, 100, 150, 200, 250], last_epoch=-1)
        for epoch in range(last_epoch, epochs):
            print("Epoch {}/{}".format(epoch + 1, epochs))
            print(scheduler.get_last_lr())
            step = 0
            # tr_loss_da = 0
            tr_loss_fr = 0
            # tr_loss_fe = 0
            for i, batch in enumerate(
                tqdm(train_set, desc="Large-training(Tr+Dev) BATCH_NUMS")
            ):
                step += 1
                distributed_model.zero_grad()
                X, Y, WCN = batch
                X_TextEmb = X[0]  # contains textual embeddings
                X_AcousticEmb = X[1]  # contains acoustic embedings
                X_masks = X[2]  # contains attention masks
                # Y_dialogueActs = torch.squeeze(Y['dialogue_act'])
                Y_frames = torch.squeeze(Y["frame"])
                Y_frames = self.get_single_label_vector(
                    Y_frames
                )  # TODO FIXME remove at certain point
                # Y_frameElements = torch.squeeze(Y['frame_element'])
                if self.use_cuda:
                    X_TextEmb = X_TextEmb.cuda()
                    X_AcousticEmb = X_AcousticEmb.cuda()
                    X_masks = X_masks.cuda()
                    # Y_dialogueActs = Y_dialogueActs.cuda()
                    Y_frames = Y_frames.cuda()
                    # Y_frameElements = Y_frameElements.cuda()
                ##FORWARD PASS
                logits_fr = distributed_model(X_TextEmb, X_AcousticEmb)
                loss_fr = self.model.slu_single_task_loss(logits_fr, Y_frames)
                tr_loss_fr += loss_fr.item()
                ## BACKWARD PASS
                loss = loss_fr
                loss.backward()
                ## UPDATING MODEL
                optimizer.step()
            ave_loss_fr = tr_loss_fr * batch_size / len(train_set.dataset)
            if epoch == 1 or epoch % 2 == 0:
                print(
                    "\t {} Epoch: {}, TRAINING loss_INTENT: {:.3f} accumulated LOSS: {:.3f}".format(
                        datetime.datetime.now(),
                        epoch + 1,
                        ave_loss_fr,
                        loss.item() * batch_size,
                    )
                )
                ##Checkpointing the model
                self.checkpoint_model(
                    "checkpoint_MULTMOD_LargeTrain.pt",
                    distributed_model.module.state_dict(),
                    epoch,
                    optimizer.state_dict(),
                    0,
                    0,
                    0,
                    0,
                )
        ## Working with the test set
        test_y, prediction = self.predict_cross_modal_validation(
            eval_set, batch_size=hyper_params["batch_size"]
        )
        f1 = f1_score(test_y, prediction, average="micro")
        print(classification_report(test_y, prediction))
        predictions_file = os.path.join(
            predictions_folder, self.network + "_predictions" + ".json"
        )
        test_x = self.dataset_obj.get_textual_sentences_from(partition="test")
        y = dict()
        y["intent"] = (test_y, prediction)
        # FIXME need to decode the labels , right now the prediction file contains only numbers
        self.print_predictions(x=test_x, y=y, predictions_file=predictions_file)

    """This version of the tune method works for the original hermit architecture for predicting DA, intent and slots"""

    def tune_text_only(
        self, train_set, val_set, epochs, patience, hyper_params, fold=1, n_folds=1
    ):
        print("\n=== Tuning TEXT_ONLY network ===")
        best_f1 = 0
        best_hyper_params = None
        best_epoch = 0
        ## getting validation set labes
        val_y = get_y_true_generator(val_set)
        batch_size = hyper_params[0]["batch_size"]
        for setting_counter, current_hyper_params in enumerate(hyper_params):
            best_hyper_params = current_hyper_params
            print("\nCurrent hyper-params: {}".format(current_hyper_params))
            rounds_without_improvements = 0
            local_best_f1 = 0

            ##checkpoinitng
            # Creating folder for checkpoints
            self.checkpoint_path = os.path.join(
                "resources", self.checkpoint_path, "checkpoints"
            )
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)

            # Creating network
            self.create_network(current_hyper_params)
            assert self.model is not None

            ##Checking if checkpoint exists
            last_epoch = 0
            if os.path.exists(os.path.join(self.checkpoint_path, "checkpoint_TXT.pt")):
                print("\t ##### Restaring from CHECKPOINT")
                (
                    state_dict,
                    last_epoch,
                    optimizer_state_dict,
                    rounds_without_improvements,
                    best_epoch,
                    best_f1,
                    local_best_f1,
                ) = self.load_checkpoint(chkpt_name="checkpoint_TXT.pt")
                self.model.load_state_dict(state_dict)

            # PyTorch
            self.model.train()

            # DDP makes sure the model is the same across all devices
            distributed_model = DistributedDataParallel(
                module=self.model.cuda(),
                process_group=get_global_group()  # ,
                # find_unused_parameters=True
            )

            # optimizer =AdamW(self.model.parameters(), weight_decay=0.01)
            optimizer = AdamW(
                distributed_model.parameters(), weight_decay=0.01
            )  # RMSprop(self.model.parameters(), weight_decay=0.000001)
            # if starting from checkpoint, load optimizer values
            if os.path.exists(os.path.join(self.checkpoint_path, "checkpoint_TXT.pt")):
                optimizer.load_state_dict(optimizer_state_dict)

            scheduler = MultiStepLR(
                optimizer, [20, 50, 100, 150, 200, 250], last_epoch=-1
            )
            ## Running through the epochs
            for epoch in range(last_epoch, epochs):
                print(
                    "Epoch {}/{} (Patience {}/{}) - Setting {}/{} - Fold {}/{}".format(
                        epoch + 1,
                        epochs,
                        rounds_without_improvements,
                        patience,
                        setting_counter + 1,
                        len(hyper_params),
                        fold,
                        n_folds,
                    )
                )
                print(scheduler.get_last_lr())
                # Sampler object --
                self.dataset_obj.get_train_sampler().set_epoch(epoch)
                step = 0
                tr_loss_da = 0
                tr_loss_fr = 0
                tr_loss_fe = 0
                for i, batch in enumerate(tqdm(train_set, desc="Training BATCH_NUMS")):
                    step += 1
                    distributed_model.zero_grad()
                    X, Y, WCN = batch
                    X_TextEmb = X[0]  # contains textual embeddings
                    X_AcousticEmb = X[1]  # contains acoustic embedings
                    X_masks = X[2]  # contains attention masks
                    Y_dialogueActs = torch.squeeze(Y["dialogue_act"])
                    Y_frames = torch.squeeze(Y["frame"])
                    Y_frameElements = torch.squeeze(Y["frame_element"])

                    assert self.use_cuda
                    if self.use_cuda:
                        X_TextEmb = X_TextEmb.cuda()
                        # X_AcousticEmb = X_AcousticEmb.cuda()
                        X_masks = X_masks.cuda()
                        Y_dialogueActs = Y_dialogueActs.cuda()
                        Y_frames = Y_frames.cuda()
                        Y_frameElements = Y_frameElements.cuda()

                    ## FORWARD PASS
                    logits_da, logits_fr, logits_fe = distributed_model(X_TextEmb)
                    ## LOSS COMPUTATION
                    loss_da, loss_fr, loss_fe = self.model.slu_loss(
                        logits_da,
                        logits_fr,
                        logits_fe,
                        Y_dialogueActs,
                        Y_frames,
                        Y_frameElements,
                        mask=X_masks,
                    )

                    tr_loss_da += loss_da.item()
                    tr_loss_fr += loss_fr.item()
                    tr_loss_fe += loss_fe.item()
                    loss = loss_da + loss_fr + loss_fe
                    ## BACKWARD PASS
                    loss.backward()
                    ## UPDATING MODEL
                    optimizer.step()

                ave_loss_da = tr_loss_da * batch_size / len(train_set.dataset)
                ave_loss_fr = tr_loss_fr * batch_size / len(train_set.dataset)
                ave_loss_fe = tr_loss_fe * batch_size / len(train_set.dataset)

                if epoch == 1 or epoch % 10 == 0:
                    print(
                        "\t {} Epoch: {}, TRAINING loss_DA: {:.3f}, loss_INTENT: {:.3f}, loss_SLOTS: {:.3f}, accumulated LOSS: {:.3f}".format(
                            datetime.datetime.now(),
                            epoch + 1,
                            ave_loss_da,
                            ave_loss_fr,
                            ave_loss_fe,
                            loss.item() * batch_size,
                        )
                    )

                ### Working with the validation set
                prediction = self.predict_validation(
                    val_set, batch_size=hyper_params[0]["batch_size"]
                )
                y = unpadv2(val_y, prediction, self.label_encoders)
                del prediction
                report = seqs_classification_report(y)
                del y
                total_f1 = report[current_hyper_params["monitor_m"]]["total"]

                if total_f1 >= best_f1:
                    best_hyper_params = current_hyper_params
                    best_f1 = total_f1
                    best_epoch = epoch
                if total_f1 >= local_best_f1:
                    local_best_f1 = total_f1
                    rounds_without_improvements = 0
                    for metric in report:
                        for label in report[metric]:
                            print(
                                " - {} {}: {}".format(
                                    metric, label, report[metric][label]
                                )
                            ),
                        print("")
                else:
                    rounds_without_improvements += 1
                if rounds_without_improvements == patience:
                    break
                ##SAVING CHECK POINT
                if epoch == 2 or epoch % 2 == 0:
                    ##Checkpointing the model
                    self.checkpoint_model(
                        "checkpoint_TXT.pt",
                        distributed_model.module.state_dict(),
                        epoch,
                        optimizer.state_dict(),
                        rounds_without_improvements,
                        best_epoch,
                        best_f1,
                        local_best_f1,
                    )
                del report
                scheduler.step()
            del local_best_f1
            del rounds_without_improvements
            del self.model
        del val_y
        print("Best F1: {}".format(best_f1))
        print("Best epoch: {}".format(best_epoch + 1))
        print("Best hyper-params: {}".format(best_hyper_params))
        return best_hyper_params, best_epoch + 1

    """This methos is for training usign the large dataset, using the original HERMIT ideas, for DA, Intents, and Slots"""

    def train_text_only(
        self, train_set, eval_set, epochs, hyper_params, predictions_folder
    ):
        print("\n=== Training TEXT_ONLY network ===")
        batch_size = hyper_params["batch_size"]
        self.create_network(hyper_params)
        assert self.model is not None
        ##Checking if checkpoint exists
        last_epoch = 0
        if os.path.exists(
            os.path.join(self.checkpoint_path, "checkpoint_TXT_LargeTrain.pt")
        ):
            print(" ## Restaring from CHECKPOINT")
            (
                state_dict,
                last_epoch,
                optimizer_state_dict,
                _,
                _,
                _,
                _,
            ) = self.load_checkpoint(chkpt_name="checkpoint_TXT_LargeTrain.pt")
            self.model.load_state_dict(state_dict)
        self.model.train()
        # DDP makes sure the model is the same across all devices
        distributed_model = DistributedDataParallel(
            module=self.model.cuda(),
            process_group=get_global_group(),
        )

        optimizer = AdamW(distributed_model.parameters(), weight_decay=0.01)
        # if starting from checkpoint, load optimizer values
        if os.path.exists(
            os.path.join(self.checkpoint_path, "checkpoint_TXT_LargeTrain.pt")
        ):
            optimizer.load_state_dict(optimizer_state_dict)
        scheduler = MultiStepLR(optimizer, [20, 50, 100, 150, 200, 250], last_epoch=-1)

        for epoch in range(last_epoch, epochs):
            print("Epoch {}/{}".format(epoch + 1, epochs))
            print(scheduler.get_last_lr())
            self.dataset_obj.get_largetrain_sampler().set_epoch(epoch)
            step = 0
            tr_loss_da = 0
            tr_loss_fr = 0
            tr_loss_fe = 0
            for i, batch in enumerate(
                tqdm(train_set, desc="Large-training(Tr+Dev) BATCH_NUMS")
            ):
                step += 1
                distributed_model.zero_grad()
                X, Y, WCN = batch
                X_TextEmb = X[0]  # contains textual embeddings
                X_AcousticEmb = X[1]  # contains acoustic embedings
                X_masks = X[2]  # contains attention masks
                Y_dialogueActs = torch.squeeze(Y["dialogue_act"])
                Y_frames = torch.squeeze(Y["frame"])
                Y_frameElements = torch.squeeze(Y["frame_element"])
                assert self.use_cuda
                if self.use_cuda:
                    X_TextEmb = X_TextEmb.cuda()
                    # X_AcousticEmb = X_AcousticEmb.cuda()
                    X_masks = X_masks.cuda()
                    Y_dialogueActs = Y_dialogueActs.cuda()
                    Y_frames = Y_frames.cuda()
                    Y_frameElements = Y_frameElements.cuda()

                ## FORWARD PASS
                logits_da, logits_fr, logits_fe = distributed_model(X_TextEmb)
                ## LOSS COMPUTATION
                loss_da, loss_fr, loss_fe = self.model.slu_loss(
                    logits_da,
                    logits_fr,
                    logits_fe,
                    Y_dialogueActs,
                    Y_frames,
                    Y_frameElements,
                    mask=X_masks,
                )

                tr_loss_da += loss_da.item()
                tr_loss_fr += loss_fr.item()
                tr_loss_fe += loss_fe.item()

                loss = loss_da + loss_fr + loss_fe
                ## BACKWARD PASS
                loss.backward()
                ## UPDATING MODEL
                optimizer.step()

            ave_loss_da = tr_loss_da * batch_size / len(train_set.dataset)
            ave_loss_fr = tr_loss_fr * batch_size / len(train_set.dataset)
            ave_loss_fe = tr_loss_fe * batch_size / len(train_set.dataset)
            if epoch == 1 or epoch % 2 == 0:
                print(
                    "\t {} Epoch: {}, TRAINING loss_DA: {:.3f}, loss_INTENT: {:.3f}, loss_SLOTS: {:.3f}, accumulated LOSS: {:.3f}".format(
                        datetime.datetime.now(),
                        epoch + 1,
                        ave_loss_da,
                        ave_loss_fr,
                        ave_loss_fe,
                        loss.item() * batch_size,
                    )
                )
                ##Checkpointing the model
                self.checkpoint_model(
                    "checkpoint_TXT_LargeTrain.pt",
                    distributed_model.module.state_dict(),
                    epoch,
                    optimizer.state_dict(),
                    0,
                    0,
                    0,
                    0,
                )
        ## Working with the test set
        prediction = self.predict_validation(
            eval_set, batch_size=hyper_params["batch_size"]
        )

        test_y = get_y_true_generator(eval_set)
        y = unpadv2(test_y, prediction, self.label_encoders)
        report = seqs_classification_report(y)
        predictions_file = os.path.join(
            predictions_folder, self.network + "_predictions" + ".json"
        )

        test_x = self.dataset_obj.get_textual_sentences_from(partition="test")
        self.print_predictions(
            x=test_x, y=y, predictions_file=predictions_file
        )  # FIXME  need to iterate in the data in a different maner, and keep the tokens, or retrieve the tokens
        del test_y
        del prediction
        for metrics in report:
            try:
                self.evaluation_report[metrics]
            except KeyError:
                self.evaluation_report[metrics] = OrderedDict()
            for label in report[metrics]:
                try:
                    self.evaluation_report[metrics][label]
                except KeyError:
                    self.evaluation_report[metrics][label] = []
        for metrics in report:
            for label in report[metrics]:
                self.evaluation_report[metrics][label].append(report[metrics][label])
        self.print_evaluation_report(out_file="/tmp/partial_results.log")
        del report
        del self.model
        return None

    """This version of the tune module works for the WCN+BERT experiments"""

    def tune_lattice_wcn(
        self, train_set, val_set, epochs, patience, hyper_params, fold=1, n_folds=1
    ):
        print("\n=== Tuning WCN network ===")
        best_f1 = 0
        best_hyper_params = None
        best_epoch = 0
        ## getting validation set labes
        val_y = get_y_true_generator(val_set)
        batch_size = hyper_params[0]["batch_size"]
        for setting_counter, current_hyper_params in enumerate(hyper_params):
            best_hyper_params = current_hyper_params
            print("\nCurrent hyper-params: {}".format(current_hyper_params))
            rounds_without_improvements = 0
            local_best_f1 = 0

            ##checkpointing
            # Creating folder for checkpoints
            self.checkpoint_path = os.path.join(
                "resources", self.checkpoint_path, "checkpoints"
            )
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)

            ##Creating network
            self.create_network(current_hyper_params)
            assert self.model is not None

            ##Checking if checkpoint exists
            last_epoch = 0
            if os.path.exists(
                os.path.join(self.checkpoint_path, "checkpoint_lattice_wcn.pt")
            ):
                print("\t ##### Restaring from CHECKPOINT")
                (
                    state_dict,
                    last_epoch,
                    optimizer_state_dict,
                    rounds_without_improvements,
                    best_epoch,
                    best_f1,
                    local_best_f1,
                ) = self.load_checkpoint(chkpt_name="checkpoint_lattice_wcn.pt")
                self.model.load_state_dict(state_dict)

            ##PyTorch
            self.model.train()

            # DDP makes sure the model is the same across all devices
            distributed_model = DistributedDataParallel(
                module=self.model.cuda(),
                process_group=get_global_group(),
                find_unused_parameters=True,
            )
            # optimizer = RMSprop(self.model.parameters(), weight_decay=0.000001)
            optimizer = AdamW(
                distributed_model.parameters(), weight_decay=0.01
            )  # RMSprop(self.model.parameters(), weight_decay=0.000001)
            # if starting from checkpoint, load optimizer values
            if os.path.exists(
                os.path.join(self.checkpoint_path, "checkpoint_lattice_wcn.pt")
            ):
                optimizer.load_state_dict(optimizer_state_dict)
            scheduler = MultiStepLR(
                optimizer, [20, 50, 100, 150, 200, 250], last_epoch=-1
            )

            for epoch in range(last_epoch, epochs):
                print(
                    "Epoch {}/{} (Patience {}/{}) - Setting {}/{} - Fold {}/{}".format(
                        epoch + 1,
                        epochs,
                        rounds_without_improvements,
                        patience,
                        setting_counter + 1,
                        len(hyper_params),
                        fold,
                        n_folds,
                    )
                )
                print(scheduler.get_last_lr())
                step = 0
                # tr_loss_da = 0
                tr_loss_fr = 0
                # tr_loss_fe = 0
                for i, batch in enumerate(tqdm(train_set, desc="Training BATCH_NUMS")):
                    step += 1
                    distributed_model.zero_grad()

                    X, Y, WCN = batch
                    # X_TextEmb = X[0] #contains textual embeddings
                    # X_AcousticEmb = X[1] #contains acoustic embedings
                    # X_masks = X[2]  # contains attention masks
                    #
                    # Y_dialogueActs = torch.squeeze(Y['dialogue_act'])
                    # Y_dialogueActs = self.get_single_label_vector(Y_dialogueActs)
                    Y_frames = torch.squeeze(Y["frame"])
                    Y_frames = self.get_single_label_vector(
                        Y_frames
                    )  # This is necessary to get the single labels
                    # Y_frameElements = torch.squeeze(Y['frame_element'])
                    inputs = {}
                    raw_in = WCN[0]  ## contains the raw inputs
                    batch_pos = WCN[1]  ## contains the positions vector
                    batch_scores = WCN[2]  # contains the wcn scores
                    raw_lens = [
                        len(utt) for utt in raw_in
                    ]  ##self.sentence_lenght  FIXME I modified this to preserve same size all the time,
                    if self.use_cuda:
                        pretrained_inputs = prepare_inputs_for_bert_xlnet(
                            raw_in,
                            raw_lens,
                            self.tokenizer,
                            batch_pos,
                            batch_scores,
                            cls_token_at_end=False,
                            cls_token="[CLS]",
                            sep_token="[SEP]",
                            cls_token_segment_id=0,
                            pad_on_left=False,
                            pad_token_segment_id=0,
                            device=torch.device("cuda"),
                        )
                        # X_TextEmb = X_TextEmb.cuda()
                        # X_AcousticEmb = X_AcousticEmb.cuda()
                        # X_masks = X_masks.cuda()
                        # Y_dialogueActs = Y_dialogueActs.cuda()
                        Y_frames = Y_frames.cuda()
                        # Y_frameElements = Y_frameElements.cuda()

                    inputs["pretrained_inputs"] = pretrained_inputs
                    masks = prepare_mask(pretrained_inputs)
                    ##FORWARD PASS
                    logits_fr = distributed_model(inputs, masks)
                    loss_fr = self.model.slu_single_task_loss(logits_fr, Y_frames)
                    tr_loss_fr += loss_fr.item()

                    ## BACKWARD PASS
                    loss = loss_fr
                    loss.backward()
                    ## UPDATING MODEL
                    optimizer.step()

                ave_loss_da = tr_loss_fr * batch_size / len(train_set.dataset)
                if epoch == 1 or epoch % 10 == 0:
                    print(
                        "\t {} Epoch: {}, TRAINING loss_INTENT: {:.3f} accumulated LOSS: {:.3f}".format(
                            datetime.datetime.now(),
                            epoch + 1,
                            ave_loss_da,
                            loss.item() * batch_size,
                        )
                    )

                ### Working with the validation set
                true_y, prediction = self.predict_validation_wcn(
                    val_set, batch_size=hyper_params[0]["batch_size"]
                )
                f1 = f1_score(true_y, prediction, average="micro")
                total_f1 = f1  # report[current_hyper_params['monitor_m']]['total']

                if total_f1 >= best_f1:
                    best_hyper_params = current_hyper_params
                    best_f1 = total_f1
                    best_epoch = epoch
                if total_f1 >= local_best_f1:
                    local_best_f1 = total_f1
                    rounds_without_improvements = 0
                    print(classification_report(true_y, prediction)),
                    print("")
                    # print(confusion_matrix(true_y,prediction))
                else:
                    rounds_without_improvements += 1
                if rounds_without_improvements == patience:
                    break
                ##SAVING CHECK POINT
                if epoch == 2 or epoch % 2 == 0:
                    ##Checkpointing the model
                    self.checkpoint_model(
                        "checkpoint_lattice_wcn.pt",
                        distributed_model.module.state_dict(),
                        epoch,
                        optimizer.state_dict(),
                        rounds_without_improvements,
                        best_epoch,
                        best_f1,
                        local_best_f1,
                    )
                scheduler.step()
            del local_best_f1
            del rounds_without_improvements
            del self.model
        del val_y
        print("Best F1: {}".format(best_f1))
        print("Best epoch: {}".format(best_epoch + 1))
        print("Best hyper-params: {}".format(best_hyper_params))
        return best_hyper_params, best_epoch + 1

    def train_lattice_wcn(
        self, train_set, eval_set, epochs, hyper_params, predictions_folder
    ):
        ## This is for the CROSS_MODAL transformer approach
        print("\n=== Training Lattice_WCN network ===")
        batch_size = hyper_params["batch_size"]
        self.create_network(hyper_params)
        assert self.model is not None
        ##Checking if checkpoint exists
        last_epoch = 0
        if os.path.exists(
            os.path.join(self.checkpoint_path, "checkpoint_WCN_LargeTrain.pt")
        ):
            print(" ## Restaring from CHECKPOINT")
            (
                state_dict,
                last_epoch,
                optimizer_state_dict,
                _,
                _,
                _,
                _,
            ) = self.load_checkpoint(chkpt_name="checkpoint_WCN_LargeTrain.pt")
            self.model.load_state_dict(state_dict)
        self.model.train()
        # DDP makes sure the model is the same across all devices
        distributed_model = DistributedDataParallel(
            module=self.model.cuda(),
            process_group=get_global_group(),
            find_unused_parameters=True,
        )
        optimizer = AdamW(distributed_model.parameters(), weight_decay=0.01)
        # if starting from checkpoint, load optimizer values
        if os.path.exists(
            os.path.join(self.checkpoint_path, "checkpoint_WCN_LargeTrain.pt")
        ):
            optimizer.load_state_dict(optimizer_state_dict)
        scheduler = MultiStepLR(optimizer, [20, 50, 100, 150, 200, 250], last_epoch=-1)
        for epoch in range(last_epoch, epochs):
            print("Epoch {}/{}".format(epoch + 1, epochs))
            print(scheduler.get_last_lr())
            step = 0
            # tr_loss_da = 0
            tr_loss_fr = 0
            # tr_loss_fe = 0
            for i, batch in enumerate(
                tqdm(train_set, desc="Large-training(Tr+Dev) BATCH_NUMS")
            ):
                step += 1
                distributed_model.zero_grad()
                X, Y, WCN = batch
                # X_TextEmb = X[0] #contains textual embeddings
                # X_AcousticEmb = X[1] #contains acoustic embedings
                # X_masks = X[2]  # contains attention masks
                # Y_dialogueActs = torch.squeeze(Y['dialogue_act'])
                Y_frames = torch.squeeze(Y["frame"])
                Y_frames = self.get_single_label_vector(Y_frames)
                # Y_frameElements = torch.squeeze(Y['frame_element'])
                inputs = {}
                raw_in = WCN[0]  ## contains the raw inputs
                batch_pos = WCN[1]  ## contains the positions vector
                batch_scores = WCN[2]  # contains the wcn scores
                raw_lens = [
                    len(utt) for utt in raw_in
                ]  ##self.sentence_lenght  FIXME I modified this to preserve same size all the time,
                if self.use_cuda:
                    pretrained_inputs = prepare_inputs_for_bert_xlnet(
                        raw_in,
                        raw_lens,
                        self.tokenizer,
                        batch_pos,
                        batch_scores,
                        cls_token_at_end=False,
                        cls_token="[CLS]",
                        sep_token="[SEP]",
                        cls_token_segment_id=0,
                        pad_on_left=False,
                        pad_token_segment_id=0,
                        device=torch.device("cuda"),
                    )
                    Y_frames = Y_frames.cuda()

                inputs["pretrained_inputs"] = pretrained_inputs
                masks = prepare_mask(pretrained_inputs)
                ##FORWARD PASS
                logits_fr = distributed_model(inputs, masks)
                loss_fr = self.model.slu_single_task_loss(logits_fr, Y_frames)
                tr_loss_fr += loss_fr.item()
                ## BACKWARD PASS
                loss = loss_fr
                loss.backward()
                ## UPDATING MODEL
                optimizer.step()
            ave_loss_fr = tr_loss_fr * batch_size / len(train_set.dataset)
            if epoch == 1 or epoch % 2 == 0:
                print(
                    "\t {} Epoch: {}, TRAINING loss_INTENT: {:.3f} accumulated LOSS: {:.3f}".format(
                        datetime.datetime.now(),
                        epoch + 1,
                        ave_loss_fr,
                        loss.item() * batch_size,
                    )
                )
                ##Checkpointing the model
                self.checkpoint_model(
                    "checkpoint_WCN_LargeTrain.pt",
                    distributed_model.module.state_dict(),
                    epoch,
                    optimizer.state_dict(),
                    0,
                    0,
                    0,
                    0,
                )
        ## Working with the test set
        test_y, prediction = self.predict_validation_wcn(
            eval_set, batch_size=hyper_params["batch_size"]
        )
        f1 = f1_score(test_y, prediction, average="micro")
        print(classification_report(test_y, prediction))
        predictions_file = os.path.join(
            predictions_folder, self.network + "_predictions" + ".json"
        )
        test_x = self.dataset_obj.get_textual_sentences_from(partition="test")
        y = dict()
        y["intent"] = (test_y, prediction)
        # FIXME need to decode the labels , right now the prediction file contains only numbers
        self.print_predictions(x=test_x, y=y, predictions_file=predictions_file)

    def evaluate_static_split(
        self,
        train_examples,
        dev_examples,
        test_examples,
        large_train_examples,
        epochs,
        patience,
        hyper_params,
        predictions_folder=None,
        fold_name=None,
        val_percentage=0.1,
    ):
        self.evaluation_report = OrderedDict()
        self.tuning_report = OrderedDict()
        for hyper_param in list(hyper_params)[0]:
            self.tuning_report[hyper_param] = []
        self.tuning_report["epoch"] = []

        ## Depending on the type of Network, we follow a different trainig approach
        if self.network == "SLU":
            best_hyper_params, best_epoch = self.tune_text_only(
                train_set=train_examples,
                val_set=dev_examples,
                epochs=epochs,
                patience=patience,
                hyper_params=hyper_params,
            )
            for best_hyper_param in best_hyper_params:
                self.tuning_report[best_hyper_param].append(
                    best_hyper_params[best_hyper_param]
                )
            self.tuning_report["epoch"].append(best_epoch)

            print("Training using train+dev data \t SLU approach \t Text Only")
            self.train_text_only(
                large_train_examples,
                test_examples,
                best_epoch,
                best_hyper_params,
                predictions_folder,
            )

        elif self.network == "SLU_Hybrid":
            best_hyper_params, best_epoch = self.tune_cross_modal(
                train_set=train_examples,
                val_set=dev_examples,
                epochs=epochs,
                patience=patience,
                hyper_params=hyper_params,
            )
            for best_hyper_param in best_hyper_params:
                self.tuning_report[best_hyper_param].append(
                    best_hyper_params[best_hyper_param]
                )
            self.tuning_report["epoch"].append(best_epoch)
            print(
                "Training using train+dev data \t SLU_Hybrid approach \t Cross-Modal Transformers (Acu+Txt)"
            )
            self.train_cross_modal(
                large_train_examples,
                test_examples,
                best_epoch,
                best_hyper_params,
                predictions_folder,
            )

        elif self.network == "SLU_WCN":
            best_hyper_params, best_epoch = self.tune_lattice_wcn(
                train_set=train_examples,
                val_set=dev_examples,
                epochs=epochs,
                patience=patience,
                hyper_params=hyper_params,
            )
            for best_hyper_param in best_hyper_params:
                self.tuning_report[best_hyper_param].append(
                    best_hyper_params[best_hyper_param]
                )
            self.tuning_report["epoch"].append(best_epoch)
            print(
                "Training using train+dev data \t SLU_WCN approach \t WCN Encoder (lattice)"
            )
            self.train_lattice_wcn(
                large_train_examples,
                test_examples,
                best_epoch,
                best_hyper_params,
                predictions_folder,
            )

        print("THE END  :) !!")
        return 1

    def print_evaluation_report(self, out_file="results.txt"):
        if not os.path.exists(os.path.dirname(out_file)):
            os.makedirs(os.path.dirname(out_file))
        with open(out_file, "w") as f:
            for hyper_param in self.tuning_report:
                print(hyper_param + ": " + str(self.tuning_report[hyper_param]))
                f.write(
                    hyper_param + ": " + str(self.tuning_report[hyper_param]) + "\n"
                )
            for metrics in self.evaluation_report:
                print(metrics + ":")
                f.write(metrics + ":\n")
                for label in self.evaluation_report[metrics]:
                    print(
                        "\t"
                        + label
                        + ": "
                        + str(self.evaluation_report[metrics][label])
                    )
                    f.write(
                        "\t"
                        + label
                        + ": "
                        + str(self.evaluation_report[metrics][label])
                        + "\n"
                    )
                print("")
                f.write("\n")

    def print_predictions(self, x, y, predictions_file="predictions.txt"):
        if not os.path.exists(os.path.dirname(predictions_file)):
            os.makedirs(os.path.dirname(predictions_file))
        x = map(
            lambda ex: [
                token for token in ex if (token != "[CLS]" and token != "[SEP]")
            ],
            x,
        )

        json_array = []

        for i, tokens in enumerate(x):
            example = OrderedDict()
            example["tokens"] = tokens
            for label, annotations in y.items():
                gold, pred = annotations
                example[label + "_gold"] = gold[i]
                example[label + "_pred"] = pred[i]
            json_array.append(example)

        with open(predictions_file, "w") as f:
            json.dump(json_array, f, indent=2)
