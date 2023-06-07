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
from __future__ import absolute_import, division, print_function

from collections import OrderedDict


def get_entities(seq, suffix=False):
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ["O"]]

    prev_tag = "O"
    prev_type = ""
    begin_offset = 0
    chunks = []

    for i, chunk in enumerate(seq + ["O"]):
        if suffix:
            tag = chunk[-1]
            type_ = chunk.split("-")[0]
        else:
            tag = chunk[0]
            type_ = chunk.split("-")[-1]

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i

        prev_tag = tag
        prev_type = type_

    return chunks


def end_of_chunk(prev_tag, tag, prev_type, type_):
    chunk_end = False

    if prev_tag == "E":
        chunk_end = True
    if prev_tag == "S":
        chunk_end = True
    if prev_tag == "B" and tag == "B":
        chunk_end = True
    if prev_tag == "B" and tag == "S":
        chunk_end = True
    if prev_tag == "B" and tag == "O":
        chunk_end = True
    if prev_tag == "I" and tag == "B":
        chunk_end = True
    if prev_tag == "I" and tag == "S":
        chunk_end = True
    if prev_tag == "I" and tag == "O":
        chunk_end = True
    if prev_tag != "O" and prev_tag != "." and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    chunk_start = False

    if tag == "B":
        chunk_start = True
    if tag == "S":
        chunk_start = True
    if prev_tag == "E" and tag == "E":
        chunk_start = True
    if prev_tag == "E" and tag == "I":
        chunk_start = True
    if prev_tag == "S" and tag == "E":
        chunk_start = True
    if prev_tag == "S" and tag == "I":
        chunk_start = True
    if prev_tag == "O" and tag == "E":
        chunk_start = True
    if prev_tag == "O" and tag == "I":
        chunk_start = True
    if tag != "O" and tag != "." and prev_type != type_:
        chunk_start = True

    return chunk_start


def f1_score(y_true, y_pred, suffix=False):
    true_entities = set(get_entities(y_true, suffix))
    pred_entities = set(get_entities(y_pred, suffix))

    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0

    return score


def accuracy_score(y_true, y_pred):
    tp = 0.0
    all_predictions = 0.0
    for true, pred in zip(y_true, y_pred):
        tp += sum([t == p for t, p in zip(true, pred)])
        all_predictions += len(true)
    return tp / all_predictions


def precision_score(y_true, y_pred, suffix=False):
    true_entities = set(get_entities(y_true, suffix))
    pred_entities = set(get_entities(y_pred, suffix))

    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)

    score = nb_correct / nb_pred if nb_pred > 0 else 0

    return score


def recall_score(y_true, y_pred, suffix=False):
    true_entities = set(get_entities(y_true, suffix))
    pred_entities = set(get_entities(y_pred, suffix))

    nb_correct = len(true_entities & pred_entities)
    nb_true = len(true_entities)

    score = nb_correct / nb_true if nb_true > 0 else 0

    return score


def compute_errors(y_true, y_pred):
    tp = 0.0
    fp = 0.0
    tn = 0.0
    fn = 0.0

    for true_example, pred_example in zip(y_true, y_pred):
        for true_token, pred_token in zip(true_example, pred_example):
            if true_token != "O":
                if true_token == pred_token:
                    tp += 1.0
                else:
                    fp += 1.0
            else:
                if true_token == pred_token:
                    tn += 1.0
                else:
                    fn += 1.0


def label_exact_match_score(y_true, y_pred):
    truth_list = [true == pred for true, pred in zip(y_true, y_pred)]
    return float(sum(truth_list)) / float(len(truth_list))


def exact_match_score(y):
    truth_list = []
    for label in y:
        y_true, y_pred = y[label]
        label_truth_list = [true == pred for true, pred in zip(y_true, y_pred)]
        if len(truth_list) == 0:
            truth_list = label_truth_list
        else:
            truth_list = [
                source and target
                for source, target in zip(truth_list, label_truth_list)
            ]
    return float(sum(truth_list)) / float(len(truth_list))


def seqs_classification_report(y):
    report = OrderedDict()

    f1 = OrderedDict()
    f1["total"] = 0.0
    report["f1"] = f1

    accuracy = OrderedDict()
    accuracy["total"] = 0.0
    report["accuracy"] = accuracy

    exact_match = OrderedDict()
    exact_match["total"] = 0.0
    report["exact_match"] = exact_match

    for label in y:
        y_true, y_pred = y[label]

        label_f1 = f1_score(y_true, y_pred)
        f1[label] = label_f1
        f1["total"] += label_f1

        label_accuracy = accuracy_score(y_true, y_pred)
        accuracy[label] = label_accuracy
        accuracy["total"] += label_accuracy

        exact_match[label] = label_exact_match_score(y_true, y_pred)

    f1["total"] = f1["total"] / len(y)
    accuracy["total"] = accuracy["total"] / len(y)
    exact_match["total"] = exact_match_score(y)
    return report
