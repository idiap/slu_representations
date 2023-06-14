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

"""


import os

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import ParameterGrid

import data.dataset as data
import learning.network as net


def distributed_init(world_size=2) -> None:
    """
    Code form Fairseq
    Initialization of the distributed DPP process.
    Args:
        world_size : the number of GPUs, by default 2
    """
    print(f'RANK: {os.environ["RANK"]}')
    dist.init_process_group(
        backend="nccl", rank=int(os.environ["RANK"]), world_size=world_size
    )
    # perform a dummy all-reduce to initialize the NCCL communicator
    if torch.cuda.is_available():
        dist.all_reduce(torch.zeros(1).cuda())


@hydra.main(version_base=None, config_path="./", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    """
    Main function
    Args:
        config_name: configuration file, hydra compatible
    """
    args = cfg
    print(f"\n ARGUMENTS:\n {args} \n")
    print(OmegaConf.to_yaml(cfg))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Getting the number of jobs/GPUs to be allocated
    world_size = args.num_jobs

    # Launching distributed DDP. There might be issues in th elogic if run in single GPU
    if args.distributed:
        distributed_init(world_size)
        print("\nDistributed INIT passed !!")
    else:
        print("\nRunning single GPU/CPU !!")

    if torch.cuda.is_available():
        # device = torch.device("cuda", torch.cuda.current_device())
        use_cuda = True
    else:
        # device = torch.device("cpu")
        use_cuda = False

    # Reading from config file relevant parameters
    hyper_params = dict()
    hyper_params["units"] = args.units
    hyper_params["dropout"] = args.dropout
    hyper_params["batch_size"] = args.batch_size
    hyper_params["loss"] = args.loss
    hyper_params["optimizer"] = args.optimizer
    hyper_params["monitor_m"] = args.monitor

    print(
        '\n\nRunning network architecture: "{}" \nCUDA mode: {} '.format(
            args.network, use_cuda
        )
    )
    print(
        '\n\nTrain data "{}" \nDev data "{}" \nTest Data "{}"'.format(
            args.train_set, args.dev_set, args.test_set
        )
    )

    # Creating the DataLoaders Objects
    if args.train_set is not None and args.test_set is not None:
        print("\n-->Reading the DATA<--")
        d = data.SLURP_Dataset(
            train_set_path=args.train_set,
            dev_set_path=args.dev_set,
            test_set_path=args.test_set,
            train_wcn=args.train_WCN_file,
            dev_wcn=args.dev_WCN_file,
            test_wcn=args.test_WCN_file,
            use_wcn=args.eval_wcn,
            load_acoustics=args.load_acoutics,
            is_DPP=args.distributed,
        )

        (
            train_set,
            dev_set,
            test_set,
            large_train_set,
            label_encoders,
        ) = d.generate_training_data_BERT_based(
            feature_spaces=[],
            run_folder=args.run_folder,
            network_type=args.network,
            emb_tr=args.acoustic_embeddings_train,
            emb_dev=args.acoustic_embeddings_dev,
            emb_test=args.acoustic_embeddings_test,
            emb_tr_dev=args.acoustic_embeddings_train_dev,
            bs=args.batch_size,
            use_wcn=args.eval_wcn,
            load_acoustic_info=args.load_acoutics,
            use_kaldi_embeddings=args.eval_kaldi_embeddings,
        )

    hyper_params["attention_width"] = [d.max_tokenized_length]
    hyper_params = ParameterGrid(hyper_params)

    # Creating instance of Network object
    slunet = net.Network(
        network=args.network,
        sentence_lenght=[d.max_tokenized_length],
        label_encoders=label_encoders,
        num_da_class=d.label_counter["dialogue_act"],
        num_intent_class=d.label_counter["frame"],
        num_slot_tags=d.label_counter["frame_element"],
        dataset_obj=d,
        text_dimensionality=args.text_dim,
        acoustic_dimensionality=args.acoustic_dim,
        run_folder=args.run_folder,
        units=args.units[0],
        dropout=args.dropout[0],
        use_cuda=use_cuda,
        use_wcn=args.eval_wcn,
    )

    file_name = args.network

    predictions_folder = os.path.join("resources", args.run_folder, "predictions")
    if (
        args.train_set is not None
        and args.dev_set is not None
        and args.test_set is not None
    ):
        fold_name = os.path.basename(os.path.dirname(args.train_set))
        file_name = file_name + "_" + fold_name
        print("\n --> Running NOW ...!")
        results = slunet.evaluate_static_split(
            train_examples=train_set,
            dev_examples=dev_set,
            test_examples=test_set,
            large_train_examples=large_train_set,
            epochs=args.epochs,
            patience=args.patience,
            hyper_params=hyper_params,
            predictions_folder=predictions_folder,
            fold_name=fold_name,
            val_percentage=0.1,
        )

    out_file = os.path.join("resources", args.run_folder, "results", file_name + ".txt")
    slunet.print_evaluation_report(out_file)
    print(f"Finished with status {results} ;)")


if __name__ == "__main__":
    """
    Calling main function
    Args:

    """
    main()
