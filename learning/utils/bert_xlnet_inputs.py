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

NOTICE: largely borrowed from su.zhu
"""
import itertools

import torch


def prepare_inputs_for_bert_xlnet(
    sentences,
    word_lengths,
    tokenizer,
    padded_position_ids,
    padded_scores=None,
    cls_token_at_end=False,
    pad_on_left=False,
    cls_token="[CLS]",
    sep_token="[SEP]",
    pad_token=0,
    sequence_a_segment_id=0,
    cls_token_segment_id=1,
    pad_token_segment_id=0,
    device=None,
):
    """Loads a data file into a list of `InputBatch`s
    `cls_token_at_end` define the location of the CLS token:
        - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
        - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
    `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    """ output: {
        'tokens': tokens_tensor,                # input_ids
        'segments': segments_tensor,            # token_type_ids
        'positions': positions_tensor,          # position_ids
        'scores': scores_tensor,                # scores
        'scores_scaler': scores_scaler_tensor,  # scores_scaler
        'mask': input_mask,                     # attention_mask
        'selects': selects_tensor,              # original_word_to_token_position
        'copies': copies_tensor                 # original_word_position
        }
    """

    pad_pos = 0
    pad_score = -1

    ## sentences are sorted by sentence length
    max_length_of_sentences = max(word_lengths)
    tokens = []
    position_ids = []
    if padded_scores is not None:
        scores = []
        scores_scaler = []
    segment_ids = []
    selected_indexes = []
    batch_size = len(sentences)

    for i in range(batch_size):
        ws = sentences[i]
        # ps = padded_position_ids[i].tolist()[:word_lengths[i]]  # BUG! 位置应从idx=1开始截取，因为数据处理时idx=0为CLS
        ps = padded_position_ids[i].tolist()[1 : word_lengths[0] + 1]
        if padded_scores is not None:
            # ss = padded_scores[i].tolist()[:word_lengths[i]]  # 同position
            ss = padded_scores[i].tolist()[1 : word_lengths[0] + 1]
        else:
            ss = [None] * word_lengths[0]
        selected_index = []
        ts, tok_ps, tok_ss, tok_sc = [], [], [], []
        for w, pos, score in zip(ws, ps, ss):
            if cls_token_at_end:
                selected_index.append(len(ts))
            else:
                selected_index.append(len(ts) + 1)
            tok_w = tokenizer.tokenize(w)
            ts += tok_w
            tok_ps += [pos] * len(tok_w)  # shared position
            if padded_scores is not None:
                tok_ss += [score] * len(tok_w)  # shared score
            tok_sc += [1.0 / len(tok_w)] * len(tok_w)
        ts += [sep_token]
        tok_ps += [max(tok_ps) + 1]
        if padded_scores is not None:
            tok_ss += [1.0]
        tok_sc += [1.0]
        si = [sequence_a_segment_id] * len(ts)
        if cls_token_at_end:
            ts = ts + [cls_token]
            # tok_ps = tok_ps + [max(tok_ps) + 1]  # BUG! 同上，若CLS在末尾，将原pos-1
            tok_ps = [x - 1 for x in tok_ps] + [max(tok_ps)]
            if padded_scores is not None:
                tok_ss = tok_ss + [1.0]
            tok_sc = tok_sc + [1.0]
            si = si + [cls_token_segment_id]
        else:
            ts = [cls_token] + ts
            # tok_ps = [1] + [x + 1 for x in tok_ps]  # BUG! 同上，tok_ps从2开始，不需要再加1
            tok_ps = [1] + tok_ps
            if padded_scores is not None:
                tok_ss = [1.0] + tok_ss
            tok_sc = [1.0] + tok_sc
            si = [cls_token_segment_id] + si
        tokens.append(ts)
        position_ids.append(tok_ps)
        if padded_scores is not None:
            scores.append(tok_ss)
        scores_scaler.append(tok_sc)
        segment_ids.append(si)
        selected_indexes.append(selected_index)

    token_lens = [len(tokenized_text) for tokenized_text in tokens]
    max_length_of_tokens = max(
        token_lens
    )  # max_length_of_sentences###max(token_lens)#FIXME Modified this to have 60 lenght sequences
    # if not cls_token_at_end: # bert
    #    assert max_length_of_tokens <= model_bert.config.max_position_embeddings
    padding_lengths = [
        max_length_of_tokens - len(tokenized_text) for tokenized_text in tokens
    ]
    if pad_on_left:
        input_mask = [
            [0] * padding_lengths[idx] + [1] * len(tokenized_text)
            for idx, tokenized_text in enumerate(tokens)
        ]
        indexed_tokens = [
            [pad_token] * padding_lengths[idx]
            + tokenizer.convert_tokens_to_ids(tokenized_text)
            for idx, tokenized_text in enumerate(tokens)
        ]
        padded_tok_positions = [
            [pad_pos] * padding_lengths[idx] + p for idx, p in enumerate(position_ids)
        ]
        if padded_scores is not None:
            padded_tok_scores = [
                [pad_score] * padding_lengths[idx] + s for idx, s in enumerate(scores)
            ]
            padded_tok_scores_scaler = [
                [pad_score] * padding_lengths[idx] + sc
                for idx, sc in enumerate(scores_scaler)
            ]
        segments_ids = [
            [pad_token_segment_id] * padding_lengths[idx] + si
            for idx, si in enumerate(segment_ids)
        ]
        selected_indexes = [
            [
                padding_lengths[idx] + i + idx * max_length_of_tokens
                for i in selected_index
            ]
            for idx, selected_index in enumerate(selected_indexes)
        ]
    else:
        input_mask = [
            [1] * len(tokenized_text) + [0] * padding_lengths[idx]
            for idx, tokenized_text in enumerate(tokens)
        ]
        indexed_tokens = [
            tokenizer.convert_tokens_to_ids(tokenized_text)
            + [pad_token] * padding_lengths[idx]
            for idx, tokenized_text in enumerate(tokens)
        ]
        padded_tok_positions = [
            p + [pad_pos] * padding_lengths[idx] for idx, p in enumerate(position_ids)
        ]
        if padded_scores is not None:
            padded_tok_scores = [
                s + [pad_score] * padding_lengths[idx] for idx, s in enumerate(scores)
            ]
            padded_tok_scores_scaler = [
                sc + [pad_score] * padding_lengths[idx]
                for idx, sc in enumerate(scores_scaler)
            ]
        segments_ids = [
            si + [pad_token_segment_id] * padding_lengths[idx]
            for idx, si in enumerate(segment_ids)
        ]
        selected_indexes = [
            [0 + i + idx * max_length_of_tokens for i in selected_index]
            for idx, selected_index in enumerate(selected_indexes)
        ]
    copied_indexes = [
        [i + idx * max_length_of_sentences for i in range(length)]
        for idx, length in enumerate(word_lengths)
    ]

    input_mask = torch.tensor(input_mask, dtype=torch.long, device=device)
    tokens_tensor = torch.tensor(indexed_tokens, dtype=torch.long, device=device)
    positions_tensor = torch.tensor(
        padded_tok_positions, dtype=torch.long, device=device
    )
    if padded_scores is not None:
        scores_tensor = torch.tensor(
            padded_tok_scores, dtype=torch.float, device=device
        )
        scores_scaler_tensor = torch.tensor(
            padded_tok_scores_scaler, dtype=torch.float, device=device
        )
    segments_tensor = torch.tensor(segments_ids, dtype=torch.long, device=device)
    selects_tensor = torch.tensor(
        list(itertools.chain.from_iterable(selected_indexes)),
        dtype=torch.long,
        device=device,
    )
    copies_tensor = torch.tensor(
        list(itertools.chain.from_iterable(copied_indexes)),
        dtype=torch.long,
        device=device,
    )
    if padded_scores is not None:
        return {
            "tokens": tokens_tensor,
            "token_lens": token_lens,
            "positions": positions_tensor,
            "scores": scores_tensor,
            "scores_scaler": scores_scaler_tensor,
            "segments": segments_tensor,
            "selects": selects_tensor,
            "copies": copies_tensor,
            "mask": input_mask,
        }
    else:
        return {
            "tokens": tokens_tensor,
            "token_lens": token_lens,
            "positions": positions_tensor,
            "segments": segments_tensor,
            "selects": selects_tensor,
            "copies": copies_tensor,
            "mask": input_mask,
        }
