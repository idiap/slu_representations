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

NOTICE: This code contains a PyTorch reimplemented version of the original 
        HERMIT NLU package avaiable at https://arxiv.org/abs/1910.00912
        Authors: Andrea Vanzo, Emanuele Bastianelli, Oliver Lemon
        Some of the new modifications/adaptations include using AdamW optimizer,
        and GRU layers instead of the LSTM, as well as a different
        way to implement the CRF
        
"""
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import BertModel, BertTokenizer

import learning.utils.Constants as Constants
from learning.layers.attention import SimpleSelfAttention
from learning.layers.Bert_ScoreAwareModels import BertEncoder as ScoreAwareBertEncoder
from learning.layers.CrossAttention import CrossAttention
from learning.layers.CrossModalAttn import MULTModel
from learning.layers.TimeDistributed import TimeDistributed
from learning.layers.torch_crf import CRF
from learning.utils.wcn_bin import bin_merger


class SLU_Network(torch.nn.Module):
    def __init__(
        self,
        max_sentence_lenght,
        label_encoders,
        da_class,
        intent_class,
        slot_tags,
        use_cuda,
        dropout=0.8,
        units=200,
        num_threads=0,
    ):
        super(SLU_Network, self).__init__()
        self.max_sentence_lenght = max_sentence_lenght
        TextEmbSize = 768  # BERT Embeddings
        AcousticEmbSize = 1536  # Acoustic Embeddings size

        self.use_cuda = use_cuda
        self.hidden_dim = units

        self.label_encoders = label_encoders
        self.labels = OrderedDict()
        for label in label_encoders:
            self.labels[label] = len(label_encoders[label].classes_)
        #  Number of tags/classes
        self.da_class = da_class
        self.intent_class = intent_class
        self.slot_tags = slot_tags

        # first BiLSTM receives as input the textual embeddings
        # self.first_biLSTM = nn.LSTM(TextEmbSize, hidden_size = 200 , bidirectional=True,
        # batch_first=True,dropout=0.2, num_layers=1)
        self.first_biLSTM = nn.GRU(
            TextEmbSize,
            hidden_size=200,
            bidirectional=True,
            batch_first=True,
            dropout=0.2,
            num_layers=1,
        )

        # second BiLSTM receives as input the concatenation of the textual embeddings and the output of the dropout of the DA
        # self.second_biLSTM = nn.LSTM(TextEmbSize+(self.hidden_dim*2), hidden_size = 200 , bidirectional=True, batch_first=True,
        #                          dropout=0.2, num_layers=1)
        # FIXME Uncomment next line for considering residual connections with textual embeddings
        self.second_biLSTM = nn.GRU(
            TextEmbSize + (self.hidden_dim * 2),
            hidden_size=200,
            bidirectional=True,
            batch_first=True,
            dropout=0.2,
            num_layers=1,
        )
        # FIXME--> Next line is for NOT using residual connections
        # self.second_biLSTM = nn.GRU(self.hidden_dim*2, hidden_size = 200 , bidirectional=True,
        #                          batch_first=True, dropout=0.2, num_layers=1)

        # third BiLSTM receives as input the concatenation of the textual embeddings and the output of the dropout of the FR
        # self.third_biLSTM  = nn.LSTM(TextEmbSize+(self.hidden_dim*2), hidden_size = 200 , bidirectional=True, batch_first=True,
        #                          dropout=0.2, num_layers=1)
        # FIXME Uncomment next line for considering residual connections with textual embeddings
        self.third_biLSTM = nn.GRU(
            TextEmbSize + (self.hidden_dim * 2),
            hidden_size=200,
            bidirectional=True,
            batch_first=True,
            dropout=0.2,
            num_layers=1,
        )
        # FIXME--> Next line is for NOT using residual connections
        # self.third_biLSTM = nn.GRU(self.hidden_dim*2, hidden_size = 200 , bidirectional=True,
        #                          batch_first=True, dropout=0.2, num_layers=1)

        self.self_attn1 = nn.MultiheadAttention(
            self.hidden_dim * 2, num_heads=2, dropout=0.2, batch_first=True
        )
        self.self_attn2 = nn.MultiheadAttention(
            self.hidden_dim * 2, num_heads=2, dropout=0.2, batch_first=True
        )
        self.self_attn3 = nn.MultiheadAttention(
            self.hidden_dim * 2, num_heads=2, dropout=0.2, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        ## Dense layer for Dialogue Acts
        self.linear_da = nn.Linear(self.hidden_dim * 2, self.da_class)
        self.timedist_da = TimeDistributed(self.linear_da, batch_first=True)

        ## Dense layer for Intents
        self.linear_intent = nn.Linear(self.hidden_dim * 2, self.intent_class)
        self.timedist_intent = TimeDistributed(self.linear_intent, batch_first=True)

        ## Dense layer for Slots
        self.linear_slots = nn.Linear(self.hidden_dim * 2, self.slot_tags)
        self.timedist_slots = TimeDistributed(self.linear_slots, batch_first=True)

        ## Idividual CRF for DA, INTENTS, and SLOTS
        self.crf_da = CRF(self.da_class)
        self.crf_intents = CRF(self.intent_class)
        self.crf_slots = CRF(self.slot_tags)

        self.evaluation_report = OrderedDict()
        self.tuning_report = OrderedDict()
        print("Network architecture initialized! (TEXT ONLY)")

    def forward(self, x_TextEmb):  # , x_AcEmb):
        h0, c0 = (
            Variable(torch.zeros(2, len(x_TextEmb), self.hidden_dim)),
            Variable(torch.zeros(2, len(x_TextEmb), self.hidden_dim)),
        )
        if self.use_cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()
        # fisrt level of the arcitecture
        # H, (h_n, c_n) = self.first_biLSTM(x_TextEmb, (h0, c0))
        H, h_n = self.first_biLSTM(x_TextEmb, h0)  ## NEW
        attention1, _ = self.self_attn1(H, H, H)
        shared_network1 = self.dropout1(attention1)
        logits_da = self.timedist_da(shared_network1)

        ## second level of the architecture
        h00, c00 = (
            Variable(
                torch.zeros(2, len(x_TextEmb + (self.hidden_dim * 2)), self.hidden_dim)
            ),
            Variable(
                torch.zeros(2, len(x_TextEmb + (self.hidden_dim * 2)), self.hidden_dim)
            ),
        )
        if self.use_cuda:
            h00 = h00.cuda()
            c00 = c00.cuda()
        shared_network = torch.cat(
            (x_TextEmb, attention1), dim=2
        )  # FIXME RESIDUAL CONECTIONS
        # shared_network = attention1 #FIXME-->this lines is to avoid residual conections, we only pass the attention output

        # H2, (h_n2, c_n2) = self.second_biLSTM(shared_network, (h00, c00))
        H2, h_n2 = self.second_biLSTM(shared_network, h00)  ## NEW
        attention2, _ = self.self_attn2(H2, H2, H2)
        shared_network2 = self.dropout2(attention2)
        logits_intents = self.timedist_intent(shared_network2)

        ##third level of the architecture
        shared_network = torch.cat(
            (x_TextEmb, attention2), dim=2
        )  # FIXME RESIDUAL CONECTIONS
        # shared_network = attention2 #FIXME-->this lines is to avoid residual conections, we only pass the attention output

        # H3, (h_n3, c_n3) = self.third_biLSTM(shared_network, (h00, c00))
        H3, h_n3 = self.third_biLSTM(shared_network, h00)  # NEW
        attention3, _ = self.self_attn3(H3, H3, H3)
        shared_network3 = self.dropout3(attention3)
        logits_slots = self.timedist_slots(shared_network3)

        return logits_da, logits_intents, logits_slots

    def slu_loss(
        self, logits_da, logits_fr, logits_fe, da_label, fr_label, fe_label, mask=None
    ):
        """
        Parameters:
        logits_da -->  predicted logits by the last dense layer for DialogueActs; shape [BS,Seq_len,da_num_labels]
        logits_fr -->  predictes logits by the last dense layer for the Intents; shape [BS,Seq_len,fr_num_labels]
        logits_fe -->  predicted logits by the last dense layer for the Slots ; shape [BS,Seq_len,fe_num_labels]
        da_label -->   True labels for DialogueActs ; shape [BS,Seq_len]
        fr_label -->   True Labels for Intents;; shape [BS,Seq_len]
        fe_label -->   True Labels for Slots; shape [BS,Seq_len]
        """
        ## Getting loss for Dialogue Acts
        da_mask = mask[:, 0 : logits_da.size(1)]
        da_mask = da_mask.transpose(1, 0)
        da_label = da_label[:, 0 : logits_da.size(1)]
        da_label = da_label.transpose(1, 0)
        logits_da = logits_da.transpose(1, 0)
        loss_da = -self.crf_da(logits_da, da_label, mask=da_mask)

        ## Getting loss for Intents
        fr_mask = mask[:, 0 : logits_fr.size(1)]
        fr_mask = fr_mask.transpose(1, 0)
        fr_label = fr_label[:, 0 : logits_fr.size(1)]
        fr_label = fr_label.transpose(1, 0)
        logits_fr = logits_fr.transpose(1, 0)
        loss_fr = -self.crf_intents(logits_fr, fr_label, mask=fr_mask)

        ## Getting loss for Slots
        fe_mask = mask[:, 0 : logits_fe.size(1)]
        fe_mask = fe_mask.transpose(1, 0)
        fe_label = fe_label[:, 0 : logits_fe.size(1)]
        fe_label = fe_label.transpose(1, 0)
        logits_fe = logits_fe.transpose(1, 0)
        loss_fe = -self.crf_slots(logits_fe, fe_label, mask=fe_mask)

        return loss_da, loss_fr, loss_fe

    def predict_slu_tasks(self, logits_da, logits_fr, logits_fe, mask=None):
        # getting Dialogue Acts predictions
        da_mask = mask[:, 0 : logits_da.size(1)]
        da_mask = da_mask.transpose(1, 0)
        logits_da = logits_da.transpose(1, 0)
        pred_da = self.crf_da.decode(logits_da, mask=da_mask)

        # getting Intents predictions
        fr_mask = mask[:, 0 : logits_fr.size(1)]
        fr_mask = fr_mask.transpose(1, 0)
        logits_fr = logits_fr.transpose(1, 0)
        pred_fr = self.crf_intents.decode(logits_fr, mask=fr_mask)

        # getting  predictions
        fe_mask = mask[:, 0 : logits_fe.size(1)]
        fe_mask = fe_mask.transpose(1, 0)
        logits_fe = logits_fe.transpose(1, 0)
        pred_fe = self.crf_slots.decode(logits_fe, mask=fe_mask)

        return pred_da, pred_fr, pred_fe


class SLU_Hybrid_Network(torch.nn.Module):
    def __init__(
        self,
        max_sentence_lenght,
        label_encoders,
        da_class,
        intent_class,
        slot_tags,
        use_cuda,
        TextEmbeddingsDim=768,
        AcousticEmbeddingsDim=768,
        dropout=0.8,
        units=200,
        num_threads=0,
    ):
        super(SLU_Hybrid_Network, self).__init__()
        # self.is_cuda = torch.cuda.is_available()
        ## Fixed parameters
        self.max_sentence_lenght = max_sentence_lenght
        TextEmbSize = TextEmbeddingsDim  # 768 #BERT Embeddings
        AcousticEmbSize = (
            AcousticEmbeddingsDim  # 768#1024#768 #1536 #Acoustic Embeddings size
        )
        latent_features = 300
        ### doubts
        self.use_cuda = use_cuda
        self.hidden_dim = units
        ###
        self.label_encoders = label_encoders
        self.labels = OrderedDict()
        for label in label_encoders:
            self.labels[label] = len(label_encoders[label].classes_)
        ## Number of tags/classes
        self.da_class = da_class
        self.intent_class = intent_class
        self.slot_tags = slot_tags

        self.cross_attention = MULTModel(
            TextEmbSize,
            AcousticEmbSize,
            seq_length=latent_features,
            output_dim=int(intent_class),
        )  # seq_length --> number of latent features

        self.evaluation_report = OrderedDict()
        self.tuning_report = OrderedDict()
        self.criterion = nn.CrossEntropyLoss()

        print("Network architecture initialized! (CROSS-MODAL)")

    def forward(self, x_TextEmb, x_AcEmb):
        OutputCrossAttnPred = self.cross_attention(
            x_TextEmb, x_AcEmb
        )  # Aligning speech-to-text using cross-attention

        return OutputCrossAttnPred

    def slu_single_task_loss(self, logits, label):
        loss = self.criterion(logits, label)
        return loss

    def pred_intent(self, logits_intent):
        pred_intent = torch.max(logits_intent, 1)[1]
        return pred_intent


class SLU_WCN_Network(torch.nn.Module):
    def __init__(
        self,
        max_sentence_lenght,
        label_encoders,
        da_class,
        intent_class,
        slot_tags,
        use_cuda,
        dropout=0.8,
        units=200,
        num_threads=0,
    ):
        super(SLU_WCN_Network, self).__init__()
        # self.is_cuda = torch.cuda.is_available()
        ## Fixed parameters
        self.max_sentence_lenght = max_sentence_lenght
        self.TextEmbSize = 768  # BERT Embeddings
        # self.AcousticEmbSize=768 #1536 #Acoustic Embeddings size
        # self.latent_features = 300
        self.dropout = dropout
        ### doubts
        self.use_cuda = use_cuda
        self.hidden_dim = units
        ###
        self.label_encoders = label_encoders
        self.labels = OrderedDict()
        for label in label_encoders:
            self.labels[label] = len(label_encoders[label].classes_)
        ## Number of tags/classes
        self.da_class = da_class
        self.intent_class = intent_class
        self.slot_tags = slot_tags

        # *****  variables for the WCN encoder
        self.bert_model_name = "bert-base-uncased"
        self.fix_bert_model = False
        self.bert_dropout = 0.1
        self.bert_num_layers = 2
        self.sent_repr = "bin_sa_cls"  ## taken from the original code
        self.score_util = "pp"  ##taken from original code
        # feature dimension
        self.fea_dim = self.TextEmbSize
        self.pretrained_model_class, self.tokenizer_class = BertModel, BertTokenizer
        self.tokenizer = self.tokenizer_class.from_pretrained(
            self.bert_model_name, do_lower_case=True
        )
        self.pretrained_model = self.pretrained_model_class.from_pretrained(
            self.bert_model_name, output_hidden_states=self.fix_bert_model
        )
        self.tokenizer.add_special_tokens(
            {"bos_token": Constants.BOS_WORD, "eos_token": Constants.EOS_WORD}
        )
        self.tokenizer.add_tokens(["<eps>", "<unk>"], special_tokens=True)
        self.pretrained_model.resize_token_embeddings(len(self.tokenizer))
        self.pretrained_model_opts = {
            "model": self.pretrained_model,
            "fix": self.fix_bert_model,
            "model_name": self.bert_model_name,
            "dp": self.bert_dropout,
        }
        self.bert_config = self.pretrained_model_opts["model"].config
        if self.sent_repr in ["bin_lstm", "bin_sa_cls", "tok_sa_cls"]:
            self.fea_dim *= 2
        if self.sent_repr in ["attn", "bin_sa", "bin_sa_cls", "tok_sa_cls"]:
            self.slf_attn = SimpleSelfAttention(self.TextEmbSize, self.dropout, "cuda")
        ##For encoding WCN using scores as well
        self.utt_sa_bert_encoder = ScoreAwareBertEncoder(
            self.bert_config,
            self.pretrained_model_opts,
            self.score_util,
            n_layers=self.bert_num_layers,
        )

        self.dropout_layer = nn.Dropout(self.dropout)

        # first BiLSTM receives as input the WCN-BERT embeddings
        # self.first_biLSTM = nn.LSTM(TextEmbSize, hidden_size = 200 , bidirectional=True, batch_first=True,
        #                          dropout=0.2, num_layers=1)
        # self.first_biLSTM = nn.GRU(TextEmbSize, hidden_size = 200 , bidirectional=True,
        #                          batch_first=True, dropout=0.2, num_layers=1)

        # second BiLSTM receives as input the concatenation of the textual embeddings and the output of the dropout of the DA
        # self.second_biLSTM = nn.LSTM(TextEmbSize+(self.hidden_dim*2), hidden_size = 200 , bidirectional=True, batch_first=True,
        #                          dropout=0.2, num_layers=1)
        # FIXME Uncomment next line for considering residual connections with textual embeddings
        # self.second_biLSTM = nn.GRU(TextEmbSize+(self.hidden_dim*2), hidden_size = 200 , bidirectional=True,
        #                         batch_first=True, dropout=0.2, num_layers=1)

        # self.self_attn1 = nn.MultiheadAttention(self.hidden_dim*2, num_heads = 2, dropout=0.2, batch_first=True)
        # self.self_attn2 = nn.MultiheadAttention(self.hidden_dim*2, num_heads = 2, dropout=0.2, batch_first=True)

        # self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)

        ## Dense layer for Dialogue Acts
        # self.linear_da = nn.Linear(self.hidden_dim*2, self.da_class)
        # self.timedist_da = TimeDistributed(self.linear_da, batch_first=True )

        ## Dense layer for Intents
        # self.linear_intent = nn.Linear(self.hidden_dim*2, self.intent_class)
        # self.timedist_intent = TimeDistributed(self.linear_intent, batch_first=True)

        #  Output layer
        self.out_layer_da = nn.Linear(
            self.TextEmbSize * 2, self.intent_class
        )  # FIXME self.da_class)
        # self.out_layer_intent = nn.Linear(TextEmbSize , self.intent_class)

        self.criterion_da = nn.CrossEntropyLoss()

        print("Network architecture initialized!")

    def get_sent_repr(self, enc_out, lens, src_pos, src_score, src_score_scaler):
        if self.sent_repr == "cls":
            sent_fea = enc_out[:, 0, :]  # (b, dm)
        elif self.sent_repr == "maxpool":
            sent_fea = enc_out.max(1)[0]  # (b, dm)
        elif self.sent_repr == "attn":  # self attn
            sent_fea = self.slf_attn(enc_out, lens)
        elif self.sent_repr == "bin_sa":  # bin-level self attn
            bin_outs, bin_lens = bin_merger(
                enc_out, src_pos, src_score, src_score_scaler=src_score_scaler
            )
            sent_fea = self.slf_attn(bin_outs, bin_lens)
        elif self.sent_repr == "bin_sa_cls":  # [bin-level self attn; cls]
            bin_outs, bin_lens = bin_merger(
                enc_out, src_pos, src_score, src_score_scaler=src_score_scaler
            )
            cls_outs = bin_outs[:, 0, :]  # (b, d)
            # whether include CLS when calculating self-attn (False by default)
            with_cls = False
            if with_cls:
                seq_outs = bin_outs  # (b, l', d)
                seq_lens = bin_lens
            else:
                seq_outs = bin_outs[:, 1:, :]  # (b, l'-1, d)
                seq_lens = [l - 1 for l in bin_lens]  # remove the first one
            sent_fea = torch.cat(
                [self.slf_attn(seq_outs, seq_lens), cls_outs], dim=1
            )  # (b, 2*d)
        elif self.sent_repr == "tok_sa_cls":
            cls_outs = enc_out[:, 0, :]
            seq_outs = enc_out[:, 1:, :]
            seq_lens = [l - 1 for l in lens]
            sent_fea = torch.cat([self.slf_attn(seq_outs, seq_lens), cls_outs], dim=1)
        else:
            raise RuntimeError("Wrong sent repr: %s" % (self.sent_repr))

        return sent_fea

    def encode_utt_sa_one_seq(self, inputs, masks):
        # inputs

        model_inputs_utt_sa = inputs["pretrained_inputs"]
        src_score_scaler = model_inputs_utt_sa["scores_scaler"]
        lens = model_inputs_utt_sa["token_lens"]
        self_mask = masks["self_mask"]

        src_seq, src_pos, src_score = (
            model_inputs_utt_sa["tokens"],
            model_inputs_utt_sa["positions"],
            model_inputs_utt_sa["scores"],
        )

        # encoder

        enc_out = self.utt_sa_bert_encoder(
            model_inputs_utt_sa, attention_mask=self_mask, default_pos=False
        )  # enc_out -> (b, l, dm)

        # utterance-level feature
        lin_in = self.get_sent_repr(enc_out, lens, src_pos, src_score, src_score_scaler)

        return lin_in, src_seq, src_pos, src_score, enc_out

    def forward(self, inputs, masks, return_attns=False):
        # ENCODING THE WCN
        lin_in, src_seq, src_pos, src_score, enc_out = self.encode_utt_sa_one_seq(
            inputs, masks
        )

        h0, c0 = (
            Variable(torch.zeros(2, len(enc_out), self.hidden_dim)),
            Variable(torch.zeros(2, len(enc_out), self.hidden_dim)),
        )
        if self.use_cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()

        # cls_outs = enc_out[:, 0, :]
        # logits_da = self.out_layer_da(lin_in)
        logits_da = self.out_layer_da(self.dropout_layer(lin_in))
        # outs = torch.sigmoid(logits_da)
        """HERMIT modules"""
        # fisrt level of the arcitecture
        # H, (h_n, c_n) = self.first_biLSTM(x_TextEmb, (h0, c0))
        # H, h_n = self.first_biLSTM(enc_out, h0)  ## NEW
        # attention1, _ = self.self_attn1 (H, H, H)
        # shared_network1 = self.dropout1(attention1)
        # logits_da = self.timedist_da(shared_network1)
        # logits_da = self.linear_da(shared_network1)
        return logits_da

    def slu_single_task_loss(self, logits, label):
        loss = self.criterion_da(logits, label)
        return loss

    def pred_intent(self, logits_intent):
        pred_intent = torch.max(logits_intent, 1)[1]
        return pred_intent

    def pred_dialogue_act(self, logits_da):
        outs = torch.sigmoid(logits_da)  # (b, nclass)
        pred_da = torch.max(outs, 1)[1]
        return pred_da
