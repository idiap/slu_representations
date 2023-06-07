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

NOTICE: This code has been modified and adapted from the original 
        HERMIT NLU package avaiable at https://arxiv.org/abs/1910.00912
        Authors: Andrea Vanzo, Emanuele Bastianelli, Oliver Lemon
"""

import os
import xml.etree.ElementTree as et
from collections import OrderedDict
from unicodedata import name

import kaldiio
import numpy as np
import pandas as pd
import torch
from progress.bar import Bar
from sklearn import preprocessing
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertModel,
    BertTokenizer,
    DistilBertModel,
    DistilBertTokenizerFast,
)

import learning.utils.Constants as Constants


class CustomTextAudioDataset(Dataset):
    def __init__(
        self,
        features=None,
        labels=None,
        acoustic_feats=None,
        max_length=0,
        use_wcn=False,
        load_acoustic_info=True,
        kaldi_embeddings=True,
    ):
        super(CustomTextAudioDataset, self).__init__()
        self.features = features
        self.labels = labels
        self.acoustic_features_file = acoustic_feats
        self.max_length = max_length
        self.use_wcn = use_wcn
        self.load_acoustic_info = load_acoustic_info
        self.kaldi_embeddings = kaldi_embeddings
        if self.use_wcn == True:
            self.model_class, self.tokenizer_class, self.pretrained_weights = (
                BertModel,
                BertTokenizer,
                "bert-base-uncased",
            )
            self.tokenizer = self.tokenizer_class.from_pretrained(
                self.pretrained_weights, do_lower_case=True
            )
            self.model = self.model_class.from_pretrained(self.pretrained_weights)
            self.tokenizer.add_special_tokens(
                {"bos_token": Constants.BOS_WORD, "eos_token": Constants.EOS_WORD}
            )
            self.tokenizer.add_tokens(["<eps>", "<unk>"], special_tokens=True)
            self.model.resize_token_embeddings(len(self.tokenizer))
        else:
            self.model_class, self.tokenizer_class, self.pretrained_weights = (
                DistilBertModel,
                DistilBertTokenizerFast,
                "distilbert-base-uncased",
            )
            self.tokenizer = self.tokenizer_class.from_pretrained(
                self.pretrained_weights, do_lower_case=False
            )
            self.model = self.model_class.from_pretrained(self.pretrained_weights)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        da_labels = []
        fr_labels = []
        fr_e_labels = []
        textual_feats = []
        audio_file = []
        textual_feats.append(self.features[idx][0])  # The input text
        if self.load_acoustic_info:
            audio_file.append(
                self.features[idx][2]
            )  ## corresponds to the audio files to be loaded
        if self.use_wcn == True:
            (
                in_seqs,
                pos_seqs,
                score_seqs,
                sa_seqs,
                sa_parent_seqs,
                sa_sib_seqs,
                sa_type_seqs,
                labels,
            ) = self.features[idx][3]
            cls = True
            ##vector of the positions CLS is considered at the beggining of the sequence
            batch_pos = (
                [1] * cls
                + [p + int(cls) for p in pos_seqs]
                + [0] * (self.max_length - len(pos_seqs) - 1)
            )
            batch_pos = np.array(batch_pos)  # torch.LongTensor(batch_pos)
            ##vector of the scores CLS is considered at the beggining of the sequence
            batch_score = (
                [1] * cls + score_seqs + [-1] * (self.max_length - len(score_seqs) - 1)
            )
            batch_score = np.array(batch_score)  # torch.FloatTensor(batch_score)

        da_labels.append(self.labels[idx][0])
        fr_labels.append(self.labels[idx][1])
        fr_e_labels.append(self.labels[idx][2])

        # getting the textual, acoustic embeddings for the batches
        text_embeddings, mask = self.__get_text_embedding(textual_feats)
        if self.load_acoustic_info == True:
            if self.kaldi_embeddings == True:  # True means Kaldi based embeddings
                (
                    not_found_idx,
                    acoustic_embeddings,
                ) = self.__get_batch_acoustic_embeddings(
                    audio_file, self.acoustic_features_file
                )
            else:  ## ELSE means HUBERT based Embeddings
                acoustic_embeddings = self.__get_batch_acoustic_embeddings_hubert_based(
                    audio_file, self.acoustic_features_file
                )
        else:
            acoustic_embeddings = None

        ## EMBEDDINGS
        if acoustic_embeddings != None:
            text_embeddings = text_embeddings.numpy()
            acoustic_embeddings = acoustic_embeddings.numpy()
        else:
            text_embeddings = text_embeddings.numpy()
        # LABELS
        da_labels = np.array(da_labels)
        fr_labels = np.array(fr_labels)
        fr_e_labels = np.array(fr_e_labels)
        mask = mask.numpy()
        ## BATCH OUTPUT
        X_batch = [text_embeddings, acoustic_embeddings, mask]
        if self.use_wcn == True:
            WCN_batch = [in_seqs, batch_pos, batch_score]
        else:
            WCN_batch = None
        Y_batch = {
            "dialogue_act": da_labels,
            "frame": fr_labels,
            "frame_element": fr_e_labels,
        }

        return X_batch, Y_batch, WCN_batch

    ## this method helps to eliminate those instances from the batch that don't have acoustic embeddings
    def __slice_tensor(self, tensor, indexes):
        update_index_val = 0
        tensor_copy = tensor
        for del_idx in indexes:
            index = del_idx - update_index_val
            tensor_copy = torch.cat(
                [tensor_copy[0:index, :, :], tensor_copy[index + 1 :, :, :]]
            )
            update_index_val += 1
        return tensor_copy

    def __get_text_embedding(self, text):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            is_split_into_words=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        ).to(device)
        features = self.model(**inputs)
        # retutn BERT EMbeddings and attention MASK
        return (
            features.last_hidden_state.to("cpu").detach(),
            inputs["attention_mask"].to("cpu").detach(),
        )

    def __get__scp_minibatch(self, acoustic_inputs):
        vector_output = []
        fd_dict = {}
        for ac_in in acoustic_inputs:
            utt, ark = ac_in.split(" ")
            ark_name, offset = ark.split(":")
            if ark_name not in fd_dict:
                fd_dict[ark_name] = kaldiio.open_like_kaldi(ark_name, "rb")
            mat = kaldiio.load_mat(ark, fd_dict=fd_dict)
            vector_output.append(torch.tensor(mat, dtype=torch.float32))

        assert len(acoustic_inputs) == len(
            vector_output
        ), f"ERROR reading acoustitc embeddings for key:{acoustic_inputs}"
        for ark_name, fd in fd_dict.items():
            fd.close()
        return pad_sequence(vector_output, batch_first=True)

    def __get_batch_acoustic_embeddings(self, audio_files_list, features_relation):
        df = pd.read_csv(features_relation, sep=" ", names=["utt", "ark"])
        acoustic_inputs = []
        not_found = []
        counter = 0
        try:
            rows = df.loc[df["utt"].isin(audio_files_list)]
            utt, ark = rows.iloc[0]  # we always keep the first one
            acoustic_inputs.append(" ".join([utt, ark]))
        except:
            not_found.append(counter)

        """for audio_files in audio_files_list:
            try:               
                rows = df.loc[df['utt'].isin(audio_files)]
                utt, ark = rows.iloc[0]  ## we always keep the first one 
                acoustic_inputs.append(' '.join([utt, ark]))

            except:
                not_found.append(counter)              
            counter+=1"""

        del df  # cleaning up memory
        if len(acoustic_inputs) != 0:
            embeddings = self.__get__scp_minibatch(acoustic_inputs)
            return not_found, embeddings
        else:
            return not_found, None

    def __get_batch_acoustic_embeddings_hubert_based(
        self, audio_files_list, features_relation
    ):  # this works for Bidisha's embeddings
        # for audio_files in audio_files_list:
        try:
            audio_fname = audio_files_list[
                0
            ]  # there is always one in the list, so we get it
            audio_fname = audio_fname + ".pt"
            tensor_fname = os.path.join(features_relation, audio_fname)
            embeddings = torch.load(tensor_fname, map_location=torch.device("cpu"))
            return torch.unsqueeze(embeddings, 0)
        except:
            print(
                "Problem readding Hubbert Embedding for file: {}".format(tensor_fname)
            )
            return None


class SLURP_Dataset(object):
    dataset = []
    train_set = []
    dev_set = []
    test_set = []
    embeddings = OrderedDict()
    labels = ["dialogue_act", "frame", "frame_element"]
    feature_encoders = OrderedDict()
    label_encoders = OrderedDict()
    label_counter = OrderedDict()

    encoders_path = None
    max_sentence_length = (
        0  # This valiable will contain the maximum lentgh (num_words) of the utterances
    )
    max_tokenized_length = 0  ## This valiable will contain the maximum lentgh (num_BERT_Tokens) of the utterances
    # SAMPLES FOR THE DPP
    samplerTrain = None
    samplerLargeTrain = None

    # Dictionaries for storing the WCN values
    train_WCN_dict = {}
    dev_WCN_dict = {}
    test_WCN_dict = {}

    # Tokenizer
    tokenizer = None

    def __init__(
        self,
        dataset_path=None,
        train_set_path=None,
        dev_set_path=None,
        test_set_path=None,
        train_wcn=None,
        dev_wcn=None,
        test_wcn=None,
        use_wcn=False,
        load_acoustics=False,
        is_DPP=False,
    ):
        if use_wcn == True:
            print("\nLoading Word Consensus Networks...")
            self.train_WCN_dict = self.__read_wcn_data(train_wcn)
            self.dev_WCN_dict = self.__read_wcn_data(dev_wcn)
            self.test_WCN_dict = self.__read_wcn_data(test_wcn)
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
        else:
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(
                "distilbert-base-uncased"
            )

        # LOADING TRAIN PARTITION
        for root, directories, file_names in os.walk(train_set_path):
            file_names = [fi for fi in file_names if fi.endswith(".hrc2")]
            if len(file_names) > 0:
                bar = Bar(
                    "Loading {} dataset: ".format(os.path.basename(root)),
                    max=len(file_names),
                )
                for filename in file_names:
                    bar.next()
                    huric_example = self.__import_example_with_BERT_tokenization(
                        os.path.join(root, filename),
                        is_train_set=True,
                        use_WCN=use_wcn,
                        load_acoustic_info=load_acoustics,
                    )
                    if huric_example is not None:
                        self.train_set.append(huric_example)
                del bar
            print("")
        # LOADING DEVEL PARTITION
        for root, directories, file_names in os.walk(dev_set_path):
            file_names = [fi for fi in file_names if fi.endswith(".hrc2")]
            if len(file_names) > 0:
                bar = Bar(
                    "Loading {} dataset: ".format(os.path.basename(root)),
                    max=len(file_names),
                )
                for filename in file_names:
                    bar.next()
                    huric_example = self.__import_example_with_BERT_tokenization(
                        os.path.join(root, filename),
                        is_dev_set=True,
                        use_WCN=use_wcn,
                        load_acoustic_info=load_acoustics,
                    )
                    if huric_example is not None:
                        self.dev_set.append(huric_example)
                del bar
            print("")
        # LOADING TEST PARTITION
        for root, directories, file_names in os.walk(test_set_path):
            file_names = [fi for fi in file_names if fi.endswith(".hrc2")]
            if len(file_names) > 0:
                bar = Bar(
                    "Loading {} dataset: ".format(os.path.basename(root)),
                    max=len(file_names),
                )
                # fout = open("ids.txt", "w")
                for filename in file_names:
                    bar.next()

                    huric_example = self.__import_example_with_BERT_tokenization(
                        os.path.join(root, filename),
                        is_test_set=True,
                        use_WCN=use_wcn,
                        load_acoustic_info=load_acoustics,
                    )
                    if huric_example is not None:
                        self.test_set.append(huric_example)
                del bar
            print("")
        print(
            "\nDataset size:\n\tTrain set: {} examples\n\tDevelopment set: {} examples\n\tTest set: {} examples".format(
                len(self.train_set), len(self.dev_set), len(self.test_set)
            )
        )
        # Checking if Distributed learning
        self.is_distributed = is_DPP
        print()
        if self.is_distributed == True:
            print("\n-->Running in distributed mode<--")
        self.samplerTrain = None
        self.samplerLargeTrain = None
        print(
            "\nMaximum lenght of the utterances when tokenized:{}; original:{}\n".format(
                self.max_tokenized_length, self.max_sentence_length
            )
        )

    def __read_wcn_data(self, fn):
        """
        * fn: wcn data file name
        * line format - word:parent:sibling:type ... \t<=>\tword:pos:score word:pos:score ... \t<=>\tlabel1;label2...
        * system act <=> utterance <=> labels
        """
        wcn_dict = {}
        in_seqs = []
        pos_seqs = []
        score_seqs = []
        sa_seqs = []
        sa_parent_seqs = []
        sa_sib_seqs = []
        sa_type_seqs = []
        labels = []
        with open(fn, "r") as fp:
            lines = fp.readlines()
            for line in lines:
                id, sa, inp, lbl = line.strip("\n\r").split("\t<=>\t")
                inp_list = inp.strip().split(" ")
                in_seq, pos_seq, score_seq = zip(
                    *[item.strip().split(":") for item in inp_list]
                )
                in_seqs = list(in_seq)  # .append(list(in_seq))
                pos_seqs = list(map(int, pos_seq))  # .append(list(map(int, pos_seq)))
                score_seqs = list(
                    map(float, score_seq)
                )  # .append(list(map(float, score_seq)))
                sa_list = sa.strip().split(" ")
                sa_seq, pa_seq, sib_seq, ty_seq = zip(
                    *[item.strip().split(":") for item in sa_list]
                )
                sa_seqs = list(sa_seq)  # .append(list(sa_seq))
                sa_parent_seqs = list(
                    map(int, pa_seq)
                )  # .append(list(map(int, pa_seq)))
                sa_sib_seqs = list(
                    map(int, sib_seq)
                )  # .append(list(map(int, sib_seq)))
                sa_type_seqs = list(map(int, ty_seq))  # .append(list(map(int, ty_seq)))

                if len(lbl) == 0:
                    labels = []  # .append([])
                else:
                    labels = lbl.strip().split(";")  # .append(lbl.strip().split(';'))
                if id not in wcn_dict.keys():
                    wcn_dict[id] = [
                        in_seqs,
                        pos_seqs,
                        score_seqs,
                        sa_seqs,
                        sa_parent_seqs,
                        sa_sib_seqs,
                        sa_type_seqs,
                        labels,
                    ]

        return wcn_dict

    def __get_WCN(self, id, isTrain, isDev, isTest):
        if isTrain == True:
            return self.train_WCN_dict[id]
        elif isDev == True:
            return self.dev_WCN_dict[id]
        elif isTest == True:
            return self.test_WCN_dict[id]

    ### Main method for reading the data in XML format
    def __import_example_with_BERT_tokenization(
        self,
        input_file,
        is_train_set=False,
        is_dev_set=False,
        is_test_set=False,
        use_WCN=False,
        load_acoustic_info=False,
    ):
        try:
            huric_example_xml = et.parse(input_file)
        except et.ParseError:
            print("Problems on file: {}".format(input_file))
            return None
        root = huric_example_xml.getroot()
        huric_example_id = root.attrib["id"]
        huric_example = dict()
        huric_example["id"] = huric_example_id
        if load_acoustic_info:
            huric_example["audio_file"] = root.attrib["audio_id"]
        for sentence in root.findall("sentence"):
            huric_example["sentence"] = sentence.text.encode("utf-8")
        ids_array = []
        lemmas_array = []
        pos_array = []
        sentence = []
        for token in root.findall("./tokens/token"):
            token_id = token.attrib["id"]
            ids_array.append(token_id)
            lemma = token.attrib["lemma"]
            lemmas_array.append(lemma.encode("utf-8"))
            pos = token.attrib["pos"]
            pos_array.append(pos)
            surface = token.attrib["surface"]
            sentence.append(surface.encode("utf-8"))
        # getting inputs for BERT encodding
        text = huric_example["sentence"].decode("utf-8").split(" ")

        huric_example["index"] = np.asarray(ids_array)
        huric_example["lemma"] = np.asarray(lemmas_array)
        huric_example["pos"] = np.asarray(pos_array)
        huric_example["tokens"] = np.asarray(sentence)
        sentence_length = len(sentence)
        if sentence_length > self.max_sentence_length:
            self.max_sentence_length = sentence_length
        huric_example["sentence_length"] = sentence_length

        ## creates empty arrays
        ner_annotations = np.full(sentence_length, fill_value="O", dtype="object")
        dialogue_act_annotations = np.full(
            sentence_length, fill_value="O", dtype="object"
        )
        frame_annotations = np.full(sentence_length, fill_value="O", dtype="object")
        frame_element_annotations = np.full(
            sentence_length, fill_value="O", dtype="object"
        )
        for dialogue_act in root.findall("./semantics/dialogueAct/token"):
            dialogue_act_annotations[
                int(dialogue_act.attrib["id"]) - 1
            ] = dialogue_act.attrib["value"]
        for ner in root.findall("./semantics/ner/token"):
            ner_annotations[int(ner.attrib["id"]) - 1] = ner.attrib["value"]
        for frame in root.findall("./semantics/frame/token"):
            frame_annotations[int(frame.attrib["id"]) - 1] = frame.attrib["value"]
        for frame_element in root.findall("./semantics/frame/frameElement/token"):
            frame_element_annotations[
                int(frame_element.attrib["id"]) - 1
            ] = frame_element.attrib["value"]

        # If the flag of WCN is on, we read the WNC files
        if use_WCN == True:
            huric_example["wcn"] = self.__get_WCN(
                os.path.basename(input_file), is_train_set, is_dev_set, is_test_set
            )
            (
                huric_example["dialogue_act"],
                huric_example["frame"],
                huric_example["frame_element"],
                tokenized_sentence,
            ) = self.__tokenize_and_preserve_labels_WCN(
                huric_example["wcn"][0],
                dialogue_act_annotations,
                frame_annotations,
                frame_element_annotations,
            )
            tokenized_length = len(
                tokenized_sentence
            )  # I'm adding 4 given that every sentence has <s> and <\s> tokens at the begining and the end of the sentence [CLS] and [SEP] are added during tokenization.
        else:
            huric_example["wcn"] = None
            ##This line works for HERMIT original
            (
                huric_example["ner"],
                huric_example["dialogue_act"],
                huric_example["frame"],
                huric_example["frame_element"],
                tokenized_sentence,
            ) = self.__tokenize_and_preserve_labels(
                text,
                ner_annotations,
                dialogue_act_annotations,
                frame_annotations,
                frame_element_annotations,
            )
            tokenized_length = len(tokenized_sentence)

        if tokenized_length > self.max_tokenized_length:
            self.max_tokenized_length = tokenized_length
        huric_example["tokenized_length"] = tokenized_length
        huric_example["bert_tokens"] = np.asarray(tokenized_sentence)

        return huric_example

    def __tokenize_and_preserve_labels_WCN(self, sentence, da, frame, frame_e):
        """
        Word piece tokenization makes it difficult to match word labels
        back up with individual word pieces. This is even more difficult
        when dealing with the WCN. This function tokenizes each
        word contained in the WCN, one at a time so that it is easier to preserve
        the correct number of tokens, and labels it accordign to the DA and FR tag.
        At this point we are considering that Da and FR will be extended
        to all the tokens in the WCN.
        We need to add functionality to be able to incule the slots in this implementation.
        At this moments that does not work.

        """
        tokenized_sentence = []
        da_labels = []
        frame_labels = []
        frame_e_labels = []
        tokenized_sentence.extend(["<s>"])
        da_labels.extend(["AAA_PAD"])
        frame_labels.extend(["AAA_PAD"])
        frame_e_labels.extend(["AAA_PAD"])
        for word in sentence[1:-1]:
            # Tokenize the word and count # of subwords the word is broken into
            tokenized_word = self.tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)
            # Add the tokenized word to the final tokenized word list
            tokenized_sentence.extend(tokenized_word)

            da_l = da[0].replace("B-", "")
            fr_l = frame[0].replace("B-", "")
            fr_e_l = frame_e[0]

            # Add the same label to the new list of labels `n_subwords` times
            da_labels.extend([da_l] * n_subwords)
            frame_labels.extend([fr_l] * n_subwords)
            frame_e_labels.extend(
                [fr_e_l] * n_subwords
            )  # FIXME THis is simple wrong, although we are not using these at the meoment!!

        tokenized_sentence.extend(["[</s>]"])
        da_labels.extend(["AAA_PAD"])
        frame_labels.extend(["AAA_PAD"])
        frame_e_labels.extend(["AAA_PAD"])

        return da_labels, frame_labels, frame_e_labels, tokenized_sentence

    def __tokenize_and_preserve_labels(self, sentence, ner, da, frame, frame_e):
        """
        Word piece tokenization makes it difficult to match word labels
        back up with individual word pieces. This function tokenizes each
        word one at a time so that it is easier to preserve the correct
        label for each subword. It is, of course, a bit slower in processing
        time, but it will help our model achieve higher accuracy.
        """

        tokenized_sentence = []
        ner_labels = []
        da_labels = []
        frame_labels = []
        frame_e_labels = []
        tokenized_sentence.extend(["[CLS]"])
        da_labels.extend(["AAA_PAD"])
        frame_labels.extend(["AAA_PAD"])
        frame_e_labels.extend(["AAA_PAD"])
        for word, ner_label, da_label, fr_label, fre_label in zip(
            sentence, ner, da, frame, frame_e
        ):  # text_labels):
            # Tokenize the word and count # of subwords the word is broken into
            tokenized_word = self.tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)

            # Add the tokenized word to the final tokenized word list
            tokenized_sentence.extend(tokenized_word)
            # Add the same label to the new list of labels `n_subwords` times
            if n_subwords == 1:
                ner_labels.extend([ner_label] * n_subwords)
                da_labels.extend([da_label] * n_subwords)
                frame_labels.extend([fr_label] * n_subwords)
                frame_e_labels.extend([fre_label] * n_subwords)
            else:
                # NER
                if (
                    "B" in ner_label.split("-")[0]
                ):  # this is a particular case for words at the Begining being splited, only the first
                    ner_labels.extend(
                        [ner_label]
                    )  # subword gets the B label, and the rest are considered as Inside (I) elements
                    ner_label = ner_label.replace("B-", "I-")
                    ner_labels.extend([ner_label] * (n_subwords - 1))
                else:
                    ner_labels.extend([ner_label] * n_subwords)
                # DIALOGUE ACTS
                if "B" in da_label.split("-")[0]:
                    da_labels.extend([da_label])
                    da_label = da_label.replace("B-", "I-")
                    da_labels.extend([da_label] * (n_subwords - 1))
                else:
                    da_labels.extend([da_label] * n_subwords)
                # INTENTIONS
                if "B" in fr_label.split("-")[0]:
                    frame_labels.extend([fr_label])
                    fr_label = fr_label.replace("B-", "I-")
                    frame_labels.extend([fr_label] * (n_subwords - 1))
                else:
                    frame_labels.extend([fr_label] * n_subwords)
                # SLOTS
                if "B" in fre_label.split("-")[0]:
                    frame_e_labels.extend([fre_label])
                    fre_label = fre_label.replace("B-", "I-")
                    frame_e_labels.extend([fre_label] * (n_subwords - 1))
                else:
                    frame_e_labels.extend([fre_label] * n_subwords)

        tokenized_sentence.extend(["[SEP]"])
        da_labels.extend(["AAA_PAD"])
        frame_labels.extend(["AAA_PAD"])
        frame_e_labels.extend(["AAA_PAD"])
        return ner_labels, da_labels, frame_labels, frame_e_labels, tokenized_sentence

    def __get_features_and_labels_from_set(
        self, partition=None, use_WCN=False, network_type="SLU", load_acoustics=False
    ):
        examples_feats = []
        examples_labels = []
        for sentence in partition:
            features = []
            label_vector = []
            # Adding the original sentence
            features.append(sentence["sentence"].decode("utf-8").split(" "))
            # Getting the tokenized version
            sentence_array = sentence["bert_tokens"]

            # getting tokens
            tokens = np.full(
                self.max_tokenized_length, fill_value="[PAD]", dtype="object"
            )
            sentence_array = sentence_array.tolist()
            sentence_array.reverse()
            for i in range(
                0, len(sentence_array)
            ):  # --> previous (self.max_tokenized_length - len(sentence_array), self.max_tokenized_length):
                tokens[i] = sentence_array.pop()
            tokens = np.asarray(tokens)
            features.append(tokens)
            # Getting headset audios IF required
            if load_acoustics:
                headset_audios = [
                    sentence["audio_file"]
                ]  # self.__get_headset_audiosIds(sentence['audio_file'])
                if len(headset_audios) > 1:
                    features.append(headset_audios)
                else:
                    features.append(
                        sentence["audio_file"]
                    )  # IF NO HEADSET audio, then use the far-field audio

            if use_WCN == True:
                features.append(sentence["wcn"])

            for label in self.labels:
                # PREVIOUS version
                # label_vector.append(to_categorical(pad_sequences([self.label_encoders[label].transform(sentence[label])],
                #                                                 maxlen=self.max_tokenized_length),
                #                                   num_classes=len(self.label_encoders[label].classes_)))
                #
                # This version does a categorical representation of the classes
                """label_vector.append(self.to_categorical(self.pad_sequences([self.label_encoders[label].transform(sentence[label])],
                              maxlen=self.max_tokenized_length,reverse_order=False),
                num_classes=len(self.label_encoders[label].classes_)))"""
                # This version does a non-categorical representation of clases, instead a INT number is defined for each label
                if network_type == "SLU":
                    label_vector.append(
                        self.pad_sequences(
                            [self.label_encoders[label].transform(sentence[label])],
                            maxlen=self.max_tokenized_length,
                            pad_value=self.label_encoders[label]
                            .transform(["AAA_PAD"])
                            .item(),
                            reverse_order=False,
                        )
                    )
                else:
                    if label == "frame" or label == "dialogue_act":
                        tags = []
                        for t in sentence[label]:
                            if t != "AAA_PAD":
                                t = t.replace("B-", "")
                                t = t.replace("I-", "")
                                tags.append(t)
                        label_vector.append(
                            self.pad_sequences(
                                [self.label_encoders[label].transform(tags)],
                                maxlen=self.max_tokenized_length,
                                pad_value=self.label_encoders[label]
                                .transform(["AAA_PAD"])
                                .item(),
                                reverse_order=False,
                            )
                        )

                    else:
                        label_vector.append(
                            self.pad_sequences(
                                [self.label_encoders[label].transform(sentence[label])],
                                maxlen=self.max_tokenized_length,
                                pad_value=self.label_encoders[label]
                                .transform(["AAA_PAD"])
                                .item(),
                                reverse_order=False,
                            )
                        )

            examples_feats.append(
                features
            )  # doing np.asarray(features)) generated a deprecation warning
            examples_labels.append(label_vector)
        return examples_feats, examples_labels

    def __my_collate_batch(self, batch, use_wcn, load_acoustics):
        # print("Size of the batch before collating {}".format(len(batch)))
        text_embeddings = []
        acoustic_embeddings = []
        masks = []
        dialogue_act_classes = []
        intent_classes = []
        slots_classes = []
        # Added for WCN process
        raw_seqs = []
        batch_pos = []
        batch_scores = []
        for instance in batch:
            # instance[0][0] --> textual embeddings
            # instance[0][1] --> acoustic embeddings
            # instance[0][2] --> attention mask
            text_embeddings.append(torch.tensor(instance[0][0]))
            masks.append(torch.tensor(instance[0][2]))
            if (
                load_acoustics and instance[0][1] is not None
            ):  ## Checking if the acoustic embedding compoment in not None
                acoustic_embeddings.append(
                    torch.Tensor.squeeze_(torch.tensor(instance[0][1]))
                )
            dialogue_act_classes.append(instance[1]["dialogue_act"])
            intent_classes.append(instance[1]["frame"])
            slots_classes.append(instance[1]["frame_element"])
            # commented for WCN
            if use_wcn == True:
                raw_seqs.append(instance[2][0])
                batch_pos.append(instance[2][1])
                batch_scores.append(instance[2][2])
        textual_embeddings = torch.cat(text_embeddings)
        if load_acoustics:
            audio_embeddings = pad_sequence(acoustic_embeddings, batch_first=True)
        else:
            audio_embeddings = None
        list_of_masks = torch.cat(masks)
        # commented for WCN

        if use_wcn == True:
            batch_pos = torch.LongTensor(np.array(batch_pos))
            batch_scores = torch.FloatTensor(np.array(batch_scores))
        da_labels = torch.LongTensor(np.array(dialogue_act_classes))
        fr_labels = torch.LongTensor(np.array(intent_classes))
        fr_e_labels = torch.LongTensor(np.array(slots_classes))

        X_batch = [textual_embeddings, audio_embeddings, list_of_masks]
        if use_wcn == True:
            WCN_batch = [list(raw_seqs), np.array(batch_pos), np.array(batch_scores)]
        else:
            WCN_batch = (
                None  # [list(raw_seqs),np.array(batch_pos),np.array(batch_scores)]
            )
        Y_batch = {
            "dialogue_act": da_labels,
            "frame": fr_labels,
            "frame_element": fr_e_labels,
        }

        return X_batch, Y_batch, WCN_batch

    def generate_training_data_BERT_based(
        self,
        feature_spaces=None,
        run_folder=None,
        network_type="SLU",
        emb_tr=None,
        emb_dev=None,
        emb_test=None,
        emb_tr_dev=None,
        bs=8,
        use_wcn=False,
        load_acoustic_info=False,
        use_kaldi_embeddings=False,
    ):
        examples = []

        self.encoders_path = os.path.join("resources", run_folder, "encoders")
        if not os.path.exists(self.encoders_path):
            os.makedirs(self.encoders_path)

        if (
            network_type == "SLU"
        ):  # The SLU architecture correspond to the original HERMIT
            self.generate_label_encoders_for_seqs(save_encoders=True)
        else:
            #  Enters here if network_type == SLU_Hybrid or SLU_WCN
            self.generate_label_encoders_for_single_out(save_encoders=True)
            # self.generate_label_encoders(save_encoders=True)

        # TRAINING DATA
        train_feats, train_labels = self.__get_features_and_labels_from_set(
            self.train_set, use_wcn, network_type, load_acoustic_info
        )
        acoustic_feats_train = emb_tr

        TAD_train = CustomTextAudioDataset(
            features=train_feats,
            labels=train_labels,
            acoustic_feats=acoustic_feats_train,
            max_length=self.max_tokenized_length,
            use_wcn=use_wcn,
            load_acoustic_info=load_acoustic_info,
            kaldi_embeddings=use_kaldi_embeddings,
        )

        if self.is_distributed:
            self.samplerTrain = torch.utils.data.distributed.DistributedSampler(
                TAD_train
            )
            world_size = torch.distributed.get_world_size()
            assert isinstance(world_size, int) and world_size > 0
            batch_size = bs[0] // world_size
        else:
            self.samplerTrain = None
            batch_size = bs[0]

        Train_DL = DataLoader(
            TAD_train,
            collate_fn=lambda batch: self.__my_collate_batch(
                batch, use_wcn, load_acoustic_info
            ),
            batch_size=batch_size,
            sampler=self.samplerTrain,
        )

        # for idx, batch in enumerate(Train_DL):
        # BATCH comes like this:
        # [ //start
        #  [[textEmbeddings][acousticembeddings][mask]]
        #  [['dialogue_act']['frame']['frame_element']]
        #  [[in_seqs] [batch_pos] [batch_score]]
        # ]//end
        # print(idx)
        # data,labels,WCN_data = batch
        #    print(labels)

        # DEVELOPMENT
        dev_feats, dev_labels = self.__get_features_and_labels_from_set(
            self.dev_set, use_wcn, network_type, load_acoustic_info
        )
        acoustic_feats_dev = emb_dev

        TAD_dev = CustomTextAudioDataset(
            features=dev_feats,
            labels=dev_labels,
            acoustic_feats=acoustic_feats_dev,
            max_length=self.max_tokenized_length,
            use_wcn=use_wcn,
            load_acoustic_info=load_acoustic_info,
            kaldi_embeddings=use_kaldi_embeddings,
        )
        Dev_DL = DataLoader(
            TAD_dev,
            collate_fn=lambda batch: self.__my_collate_batch(
                batch, use_wcn, load_acoustic_info
            ),
            batch_size=bs[0],
            shuffle=False,
        )

        # TEST
        test_feats, test_labels = self.__get_features_and_labels_from_set(
            self.test_set, use_wcn, network_type, load_acoustic_info
        )
        acoustic_feats_test = emb_test

        TAD_test = CustomTextAudioDataset(
            features=test_feats,
            labels=test_labels,
            acoustic_feats=acoustic_feats_test,
            max_length=self.max_tokenized_length,
            use_wcn=use_wcn,
            load_acoustic_info=load_acoustic_info,
            kaldi_embeddings=use_kaldi_embeddings,
        )
        Test_DL = DataLoader(
            TAD_test,
            collate_fn=lambda batch: self.__my_collate_batch(
                batch, use_wcn, load_acoustic_info
            ),
            batch_size=bs[0],
            shuffle=False,
        )

        # LARGE TRAIN DATA
        large_train_feats = train_feats + dev_feats
        large_train_labels = train_labels + dev_labels
        acoustic_feats_train_dev = emb_tr_dev

        TAD_LargeTrain = CustomTextAudioDataset(
            features=large_train_feats,
            labels=large_train_labels,
            acoustic_feats=acoustic_feats_train_dev,
            max_length=self.max_tokenized_length,
            use_wcn=use_wcn,
            load_acoustic_info=load_acoustic_info,
            kaldi_embeddings=use_kaldi_embeddings,
        )
        if self.is_distributed:
            self.samplerLargeTrain = torch.utils.data.distributed.DistributedSampler(
                TAD_LargeTrain
            )
            world_size = torch.distributed.get_world_size()
            assert isinstance(world_size, int) and world_size > 0
            batch_size = bs[0] // world_size
        else:
            self.samplerLargeTrain = None
            batch_size = bs[0]
        LargeTrain_DL = DataLoader(
            TAD_LargeTrain,
            collate_fn=lambda batch: self.__my_collate_batch(
                batch, use_wcn, load_acoustic_info
            ),
            batch_size=batch_size,
            sampler=self.samplerLargeTrain,
        )

        return Train_DL, Dev_DL, Test_DL, LargeTrain_DL, self.label_encoders

    def __get_headset_audiosIds(self, file_names):
        headsets = []
        for f in file_names:
            if "headset" in f:
                headsets.append(f)
        return headsets

    def print_sentences(self, out_file=None, include_id=False):
        if out_file:
            with open(out_file, "w") as f:
                bar = Bar("Printing dataset: ", max=len(self.dataset))

                for i, example in enumerate(self.dataset):
                    bar.next()
                    if include_id:
                        f.write(example["id"] + "\t")
                    f.write(example["sentence"] + "\n")
        else:
            bar = Bar("Printing dataset: ", max=len(self.dataset))

            for i, example in enumerate(self.dataset):
                bar.next()
                if include_id:
                    print(example["id"] + "\t"),
                print(example["sentence"])

    def generate_feature_encoders(self, feature_spaces, save_encoders=False):
        for feature_space in feature_spaces:
            # print("generate_feature_encoders")
            feature_encoder = preprocessing.LabelEncoder()

            if feature_space == "pos":
                feature_encoder.fit(self.bag_of_pos)
            elif feature_space == "ner":
                feature_encoder.fit(self.bag_of_ner)
            else:
                bag_of_features = set()
                for sentence in self.dataset:
                    for label in sentence[feature_space]:
                        bag_of_features.add(label)
                for sentence in self.train_set:
                    for label in sentence[feature_space]:
                        bag_of_features.add(label)
                for sentence in self.test_set:
                    for label in sentence[feature_space]:
                        bag_of_features.add(label)
                bag_of_features = list(bag_of_features)
                feature_encoder.fit(bag_of_features)

            self.feature_encoders[feature_space] = feature_encoder
            embedding = np.identity(len(feature_encoder.classes_), dtype="float32")
            self.embeddings[feature_space] = embedding
            if save_encoders:
                np.save(
                    os.path.join(self.encoders_path, feature_space + "_labels.npy"),
                    feature_encoder.classes_,
                )

    def generate_label_encoders_for_single_out(self, save_encoders=False):
        for label in self.labels:
            bag_of_labels = set()
            bag_of_labels.add("AAA_PAD")
            ##This IF controls the labels for DA and Intent. These are treat as clases and not sequence labeling
            if label != "frame" and label != "dialogue_act":
                for sentence in self.dataset:
                    for l in sentence[label]:
                        bag_of_labels.add(l)
                for sentence in self.train_set:
                    for l in sentence[label]:
                        bag_of_labels.add(l)
                for sentence in self.dev_set:
                    for l in sentence[label]:
                        bag_of_labels.add(l)
                for sentence in self.test_set:
                    for l in sentence[label]:
                        bag_of_labels.add(l)
            else:  # The frame elements are considered as sequence labels
                for sentence in self.train_set:
                    for l in sentence[label]:
                        if l != "AAA_PAD":
                            l = l.replace("B-", "")
                            l = l.replace("I-", "")
                            bag_of_labels.add(l)
                for sentence in self.dev_set:
                    for l in sentence[label]:
                        if l != "AAA_PAD":
                            l = l.replace("B-", "")
                            l = l.replace("I-", "")
                            bag_of_labels.add(l)
                for sentence in self.test_set:
                    for l in sentence[label]:
                        if l != "AAA_PAD":
                            l = l.replace("B-", "")
                            l = l.replace("I-", "")
                            bag_of_labels.add(l)

            bag_of_labels = list(bag_of_labels)
            self.label_counter[label] = len(bag_of_labels)
            label_encoder = preprocessing.LabelEncoder()
            label_encoder.fit(bag_of_labels)

            self.label_encoders[label] = label_encoder
            if save_encoders:
                np.save(
                    os.path.join(self.encoders_path, label + "_labels.npy"),
                    label_encoder.classes_,
                )

    def generate_label_encoders_for_seqs(self, save_encoders=False):
        for label in self.labels:
            bag_of_labels = set()
            bag_of_labels.add("AAA_PAD")
            for sentence in self.dataset:
                for l in sentence[label]:
                    bag_of_labels.add(l)
            for sentence in self.train_set:
                for l in sentence[label]:
                    bag_of_labels.add(l)
            for sentence in self.dev_set:
                for l in sentence[label]:
                    bag_of_labels.add(l)
            for sentence in self.test_set:
                for l in sentence[label]:
                    bag_of_labels.add(l)
            bag_of_labels = list(bag_of_labels)
            self.label_counter[label] = len(bag_of_labels)
            label_encoder = preprocessing.LabelEncoder()
            label_encoder.fit(bag_of_labels)

            self.label_encoders[label] = label_encoder
            if save_encoders:
                np.save(
                    os.path.join(self.encoders_path, label + "_labels.npy"),
                    label_encoder.classes_,
                )

    def get_textual_sentences_from(self, partition="test"):
        texts = []
        if partition == "test":
            for example in self.test_set:
                texts.append(example["bert_tokens"].tolist())
        return texts

    """
    These I implemented before
    """

    def pad_sequences(
        self, labels_array, maxlen=None, pad_value=-1, reverse_order=True, dtype="int32"
    ):
        padded_sequence = np.full((maxlen), pad_value, dtype=dtype)
        # padded_sequence = np.zeros(shape=(maxlen), dtype=dtype)
        if reverse_order:
            padded_sequence[-(len(labels_array[0])) :] = labels_array[0]

            # return np.roll(padded_sequence,-1)
        else:
            padded_sequence[: len(labels_array[0])] = labels_array[0]
            # return np.roll(padded_sequence, 1)
        return padded_sequence

    def to_categorical(self, y, num_classes=None, dtype="float32"):
        """1-hot encodes a tensor"""
        return np.eye(num_classes, dtype=dtype)[y]

    def compute_random_baseline(self):
        from learning.metrics.sequence_labeling import classification_report

        np.random.seed(42)
        distribution = self.statistics()
        self.generate_label_encoders()
        dialogue_act_labels = self.label_encoders["dialogue_act"].classes_
        frame_labels = self.label_encoders["frame"].classes_
        frame_element_labels = self.label_encoders["frame_element"].classes_

        dialogue_act_labels = np.delete(
            dialogue_act_labels, np.argwhere(dialogue_act_labels == "AAA_PAD")
        )
        frame_labels = np.delete(frame_labels, np.argwhere(frame_labels == "AAA_PAD"))
        frame_element_labels = np.delete(
            frame_element_labels, np.argwhere(frame_element_labels == "AAA_PAD")
        )
        dialogue_act_labels = np.delete(
            dialogue_act_labels, np.argwhere(dialogue_act_labels == "O")
        )
        frame_labels = np.delete(frame_labels, np.argwhere(frame_labels == "O"))
        frame_element_labels = np.delete(
            frame_element_labels, np.argwhere(frame_element_labels == "O")
        )

        dialogue_act_labels = list(
            set(
                [
                    label.replace("B-", "").replace("I-", "")
                    for label in dialogue_act_labels
                ]
            )
        )
        frame_labels = list(
            set([label.replace("B-", "").replace("I-", "") for label in frame_labels])
        )
        frame_element_labels = list(
            set(
                [
                    label.replace("B-", "").replace("I-", "")
                    for label in frame_element_labels
                ]
            )
        )

        dialogue_act_distribution = []
        frame_distribution = []
        frame_element_distribution = []
        for label in dialogue_act_labels:
            print(label, distribution["dialogue_act"][label])
            dialogue_act_distribution.append(distribution["dialogue_act"][label])
        for label in frame_labels:
            frame_distribution.append(distribution["frame"][label])
        for label in frame_element_labels:
            frame_element_distribution.append(distribution["frame_element"][label])

        np.random.shuffle(self.dataset)
        folds = [self.dataset[i::10] for i in range(10)]
        for i, fold in enumerate(folds):
            print("Iteration {}".format(i))
            y = OrderedDict()
            y_true = []
            y_pred = []
            for example in fold:
                new_example = []
                for token in example["dialogue_act"]:
                    if token.startswith("B-"):
                        current_label = token.split("-")[1]
                        random_label = np.random.choice(
                            dialogue_act_labels, 1, dialogue_act_distribution
                        )[0]
                        new_token = token.replace(current_label, random_label)
                    else:
                        new_token = token.replace(current_label, random_label)
                    new_example.append(new_token)
                y_true.append(example["dialogue_act"].tolist())
                y_pred.append(new_example)
            y["dialogue_act"] = (y_true, y_pred)
            y_true = []
            y_pred = []
            for example in fold:
                new_example = []
                for token in example["frame"]:
                    if token.startswith("B-"):
                        current_label = token.split("-")[1]
                        random_label = np.random.choice(
                            frame_labels, 1, frame_distribution
                        )[0]
                        new_token = token.replace(current_label, random_label)
                    else:
                        new_token = token.replace(current_label, random_label)
                    new_example.append(new_token)
                y_true.append(example["frame"].tolist())
                y_pred.append(new_example)
            y["frame"] = (y_true, y_pred)
            y_true = []
            y_pred = []

            for example in fold:
                new_example = []
                for token in example["frame_element"]:
                    if token.startswith("B-"):
                        current_label = token.split("-")[1]
                        random_label = np.random.choice(
                            frame_element_labels, 1, frame_element_distribution
                        )[0]
                        new_token = token.replace(current_label, random_label)
                    else:
                        new_token = token.replace(current_label, random_label)
                    new_example.append(new_token)
                y_true.append(example["frame_element"].tolist())
                y_pred.append(new_example)
            y["frame_element"] = (y_true, y_pred)
            print(classification_report(y))

    def statistics(self):
        avg_length_of_sentence = 0.0
        total_number_of_dialogue_act = 0
        total_number_of_frame = 0
        total_number_of_frame_element = 0
        dialogue_act_number = dict()
        frame_number = dict()
        frame_element_number = dict()
        lexical_unit_distribution = dict()
        dialogue_act_predicates = 0.0
        frame_predicates = 0.0
        frame_element_predicates = 0.0
        for example in self.dataset:
            avg_length_of_sentence += len(example["tokens"])
            for i, token in enumerate(example["tokens"]):
                if example["dialogue_act"][i].startswith("B-"):
                    total_number_of_dialogue_act += 1
                    name = example["dialogue_act"][i].split("-")[1]
                    if name not in dialogue_act_number:
                        dialogue_act_number[name] = 1
                    else:
                        dialogue_act_number[name] += 1
                    dialogue_act_predicates += 1
                if example["frame"][i].startswith("B-"):
                    total_number_of_frame += 1
                    name = example["frame"][i].split("-")[1]
                    if name not in frame_number:
                        frame_number[name] = 1
                    else:
                        frame_number[name] += 1
                    frame_predicates += 1
                if example["frame_element"][i].startswith("B-"):
                    total_number_of_frame_element += 1
                    name = example["frame_element"][i].split("-")[1]
                    if name not in frame_element_number:
                        frame_element_number[name] = 1
                    else:
                        frame_element_number[name] += 1
                    frame_element_predicates += 1
                    if name == "Lexical_unit":
                        index = example["index"][i]
                        if index not in lexical_unit_distribution:
                            lexical_unit_distribution[index] = 1
                        else:
                            lexical_unit_distribution[index] += 1
        avg_length_of_sentence = avg_length_of_sentence / len(self.dataset)
        print("Number of sentences:\t\t\t{}".format(len(self.dataset)))
        print(
            "Average length of sentence:\t\t{}".format(round(avg_length_of_sentence, 2))
        )
        print("Dialogue act label set:\t\t\t{}".format(len(dialogue_act_number)))
        print("Frame label set:\t\t\t\t{}".format(len(frame_number)))
        print("Frame element label set:\t\t{}".format(len(frame_element_number)))
        print("Total number of dialogue act:\t{}".format(total_number_of_dialogue_act))
        print("Total number of frame:\t\t\t{}".format(total_number_of_frame))
        print(
            "Total number of frame element:\t{}".format(total_number_of_frame_element)
        )
        print(
            "Average dialogue act/sentence:\t{}".format(
                round(float(total_number_of_dialogue_act) / len(self.dataset), 2)
            )
        )
        print(
            "Average frame/sentence:\t\t\t{}".format(
                round(float(total_number_of_frame) / len(self.dataset), 2)
            )
        )
        print(
            "Average frame element/sentence:\t{}".format(
                round(float(total_number_of_frame_element) / len(self.dataset), 2)
            )
        )
        print(
            "Average frame/dialogue act:\t\t{}".format(
                round(
                    float(total_number_of_frame) / float(total_number_of_dialogue_act),
                    2,
                )
            )
        )
        print(
            "Average frame element/frame:\t{}".format(
                round(
                    float(total_number_of_frame_element) / float(total_number_of_frame),
                    2,
                )
            )
        )
        print(
            "Lexical unit distribution:\t\t{}".format(
                sorted(
                    lexical_unit_distribution.items(), key=lambda x: x[1], reverse=True
                )
            )
        )

        distribution = OrderedDict()
        dialogue_act_distribution = OrderedDict()
        frame_distribution = OrderedDict()
        frame_element_distribution = OrderedDict()
        distribution["dialogue_act"] = dialogue_act_distribution
        distribution["frame"] = frame_distribution
        distribution["frame_element"] = frame_element_distribution

        print(
            "Dialogue act distribution:\t\t{}".format(
                sorted(dialogue_act_number.items(), key=lambda x: x[1], reverse=True)
            )
        )
        for label, number in dialogue_act_number.items():
            dialogue_act_distribution[label] = float(number) / dialogue_act_predicates
        print(
            "Frame distribution:\t\t\t\t{}".format(
                sorted(frame_number.items(), key=lambda x: x[1], reverse=True)
            )
        )
        for label, number in frame_number.items():
            frame_distribution[label] = float(number) / frame_predicates
        print(
            "Frame element distribution:\t\t{}".format(
                sorted(frame_element_number.items(), key=lambda x: x[1], reverse=True)
            )
        )
        for label, number in frame_element_number.items():
            frame_element_distribution[label] = float(number) / frame_element_predicates

        return distribution

    def statistics_benchmark(self):
        avg_length_of_sentence = 0.0
        total_number_of_dialogue_act = 0
        total_number_of_frame = 0
        total_number_of_intent = 0
        total_number_of_frame_element = 0
        dialogue_act_number = dict()
        frame_number = dict()
        frame_element_number = dict()
        intent_number = dict()
        lexical_unit_distribution = dict()
        dialogue_act_predicates = 0.0
        frame_predicates = 0.0
        frame_element_predicates = 0.0
        intent_predicates = 0.0
        for example in self.dataset:
            avg_length_of_sentence += len(example["tokens"])
            for i, token in enumerate(example["tokens"]):
                dialogue_act_check = False
                frame_check = False
                if example["dialogue_act"][i].startswith("B-"):
                    dialogue_act_check = True
                    total_number_of_dialogue_act += 1
                    name = example["dialogue_act"][i][2:]
                    if name not in dialogue_act_number:
                        dialogue_act_number[name] = 1
                    else:
                        dialogue_act_number[name] += 1
                    dialogue_act_predicates += 1
                if example["frame"][i].startswith("B-"):
                    frame_check = True
                    total_number_of_frame += 1
                    name = example["frame"][i][2:]
                    if name not in frame_number:
                        frame_number[name] = 1
                    else:
                        frame_number[name] += 1
                    frame_predicates += 1
                if dialogue_act_check and frame_check:
                    name = (
                        example["dialogue_act"][i][2:] + "_" + example["frame"][i][2:]
                    )
                    total_number_of_intent += 1
                    if name not in intent_number:
                        intent_number[name] = 1
                    else:
                        intent_number[name] += 1
                    intent_predicates += 1
                if example["frame_element"][i].startswith("B-"):
                    total_number_of_frame_element += 1
                    name = example["frame_element"][i][2:]
                    if name not in frame_element_number:
                        frame_element_number[name] = 1
                    else:
                        frame_element_number[name] += 1
                    frame_element_predicates += 1
                    if name == "Lexical_unit":
                        index = example["index"][i]
                        if index not in lexical_unit_distribution:
                            lexical_unit_distribution[index] = 1
                        else:
                            lexical_unit_distribution[index] += 1
        avg_length_of_sentence = avg_length_of_sentence / len(self.dataset)
        print("Number of sentences:\t\t\t{}".format(len(self.dataset)))
        print(
            "Average length of sentence:\t\t{}".format(round(avg_length_of_sentence, 2))
        )
        print("Dialogue act label set:\t\t\t{}".format(len(dialogue_act_number)))
        print("Frame label set:\t\t\t\t{}".format(len(frame_number)))
        print("Frame element label set:\t\t{}".format(len(frame_element_number)))
        print("Intent label set:\t\t\t\t{}".format(len(intent_number)))
        print("Total number of dialogue act:\t{}".format(total_number_of_dialogue_act))
        print("Total number of frame:\t\t\t{}".format(total_number_of_frame))
        print("Total number of intent:\t\t\t{}".format(total_number_of_intent))
        print(
            "Total number of frame element:\t{}".format(total_number_of_frame_element)
        )
        print(
            "Average dialogue act/sentence:\t{}".format(
                round(float(total_number_of_dialogue_act) / len(self.dataset), 2)
            )
        )
        print(
            "Average frame/sentence:\t\t\t{}".format(
                round(float(total_number_of_frame) / len(self.dataset), 2)
            )
        )
        print(
            "Average intent/sentence:\t\t\t{}".format(
                round(float(total_number_of_intent) / len(self.dataset), 2)
            )
        )
        print(
            "Average frame element/sentence:\t{}".format(
                round(float(total_number_of_frame_element) / len(self.dataset), 2)
            )
        )
        print(
            "Average frame/dialogue act:\t\t{}".format(
                round(
                    float(total_number_of_frame) / float(total_number_of_dialogue_act),
                    2,
                )
            )
        )
        print(
            "Average frame element/frame:\t{}".format(
                round(
                    float(total_number_of_frame_element) / float(total_number_of_frame),
                    2,
                )
            )
        )
        print(
            "Average frame element/intent:\t{}".format(
                round(
                    float(total_number_of_frame_element)
                    / float(total_number_of_intent),
                    2,
                )
            )
        )
        print(
            "Lexical unit distribution:\t\t{}".format(
                sorted(
                    lexical_unit_distribution.items(), key=lambda x: x[1], reverse=True
                )
            )
        )

        distribution = OrderedDict()
        dialogue_act_distribution = OrderedDict()
        frame_distribution = OrderedDict()
        frame_element_distribution = OrderedDict()
        distribution["dialogue_act"] = dialogue_act_distribution
        distribution["frame"] = frame_distribution
        distribution["frame_element"] = frame_element_distribution

        print(
            "Dialogue act distribution:\t\t{}".format(
                sorted(dialogue_act_number.items(), key=lambda x: x[1], reverse=True)
            )
        )
        for label, number in dialogue_act_number.items():
            dialogue_act_distribution[label] = float(number) / dialogue_act_predicates
        print(
            "Frame distribution:\t\t\t\t{}".format(
                sorted(frame_number.items(), key=lambda x: x[1], reverse=True)
            )
        )
        for label, number in frame_number.items():
            frame_distribution[label] = float(number) / frame_predicates
        print(
            "Frame element distribution:\t\t{}".format(
                sorted(frame_element_number.items(), key=lambda x: x[1], reverse=True)
            )
        )
        for label, number in frame_element_number.items():
            frame_element_distribution[label] = float(number) / frame_element_predicates

        return distribution

    def get_train_sampler(self):
        return self.samplerTrain

    def get_largetrain_sampler(self):
        return self.samplerLargeTrain
