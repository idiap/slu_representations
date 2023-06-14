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
from collections import OrderedDict

import numpy as np
import torch


def reshape_for_training(dataset):
    num_input = len(dataset[0][0])
    num_label = len(dataset[0][1])
    X = []
    Y = []
    for i in range(num_input):
        X.append([])
    for i in range(num_label):
        Y.append([])
    for example in dataset:
        for i in range(len(example[0])):
            X[i].append(example[0][i])
        for i in range(len(example[1])):
            Y[i].append(example[1][i])
    return X, Y


def unpadv2(gold, prediction, label_encoders=None):
    y = OrderedDict()
    for i, label in enumerate(label_encoders):
        gold_standard = np.array(gold[i])
        predicted = np.array(prediction[i])
        y_true = []
        y_pred = []
        for y_true_i, y_pred_i in zip(gold_standard, predicted):
            new_true = []
            new_pred = []
            # This loop actually works for omitting the first and last tokens of the predicted
            # vector aligned with the CLS and SEP tokens
            # temp_y_true_i = y_true_i[y_true_i!=0]
            # assert len(temp_y_true_i) == len(y_pred_i),\
            #    f"ERROR: Size of true labels and predicted labels are different"
            for true_token, pred_token in zip(y_true_i, y_pred_i):
                if true_token != 0:
                    new_true.append(true_token)
                    new_pred.append(pred_token)
            y_true.append(label_encoders[label].inverse_transform(new_true).tolist())
            y_pred.append(label_encoders[label].inverse_transform(new_pred).tolist())
            del new_true
            del new_pred
        y[label] = (y_true, y_pred)
        del y_true
        del y_pred
        del gold_standard
        del predicted
    return y


def unpad(gold, prediction, label_encoders=None):
    y = OrderedDict()
    for i, label in enumerate(label_encoders):
        gold_standard = np.argmax(gold[i], axis=2)
        predicted = np.argmax(prediction[i], axis=2)
        y_true = []
        y_pred = []
        for y_true_i, y_pred_i in zip(gold_standard, predicted):
            new_true = []
            new_pred = []
            for true_token, pred_token in zip(y_true_i, y_pred_i):
                if true_token != 0:
                    new_true.append(true_token)
                    new_pred.append(pred_token)
            y_true.append(label_encoders[label].inverse_transform(new_true).tolist())
            y_pred.append(label_encoders[label].inverse_transform(new_pred).tolist())
            del new_true
            del new_pred
        y[label] = (y_true, y_pred)
        del y_true
        del y_pred
        del gold_standard
        del predicted
    return y


def get_y_true_generator(gold_generator):
    y_true_da = []
    y_true_fr = []
    y_true_fre = []
    for x, y, wcn in gold_generator:  # remove wcn for normal hermit
        for d, f, e in zip(y["dialogue_act"], y["frame"], y["frame_element"]):
            y_true_da.append(torch.squeeze(d).tolist())
            y_true_fr.append(torch.squeeze(f).tolist())
            y_true_fre.append(torch.squeeze(e).tolist())
    return [y_true_da, y_true_fr, y_true_fre]
