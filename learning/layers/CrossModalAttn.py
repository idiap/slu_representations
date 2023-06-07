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
        Multimodal Transformer for Unaligned Multimodal Language Sequences
        (https://github.com/yaohungt/Multimodal-Transformer)
        Authors: Yao-Hung Hubert Tsai (yaohungt@cs.cmu.edu) and 
        Shaojie Bai (shaojieb@andrew.cmu.edu)

"""
import torch
import torch.nn.functional as F
from torch import nn

from learning.layers.transformer import TransformerEncoder


class MULTModel(nn.Module):
    def __init__(self, orig_d_l, orig_d_a, seq_length=300, output_dim=1):
        """
        Construct a MulT model.
        """
        super(MULTModel, self).__init__()
        self.orig_d_l = orig_d_l
        self.orig_d_a = orig_d_a  # data dimensionality
        self.d_l = seq_length
        self.d_a = seq_length  # sequence lentgh
        self.aonly = True
        self.lonly = True  # IF true, use the crossmodal fusion into l (text modality)
        self.num_heads = 5
        self.layers = 4
        self.attn_dropout = 0.1
        self.attn_dropout_a = 0.0
        self.relu_dropout = 0.1
        self.res_dropout = 0.1
        self.out_dropout = 0.0
        self.embed_dropout = 0.3
        self.attn_mask = True
        self.output_dim = output_dim

        combined_dim = 2 * seq_length  # self.d_l + self.d_a

        # self.partial_mode = self.lonly + self.aonly
        # if self.partial_mode == 1:
        #    combined_dim = 2 * self.d_l   # assuming d_l == d_a == d_v
        # else:
        #    combined_dim = 2 * (self.d_l + self.d_a)

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(
            self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False
        )
        self.proj_a = nn.Conv1d(
            self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False
        )

        # 2. Crossmodal Attentions
        if self.lonly:
            self.trans_l_with_a = self.get_network(self_type="la")

        if self.aonly:
            self.trans_a_with_l = self.get_network(self_type="al")

        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_mem = self.get_network(self_type="l_mem", layers=3)
        self.trans_a_mem = self.get_network(self_type="a_mem", layers=3)

        ## Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)

        self.proj3 = nn.Linear(combined_dim, combined_dim)
        self.proj4 = nn.Linear(combined_dim, combined_dim)

        # 4. Multihead attention to alingn audio sequences to text sequences
        self.multihead_attn = nn.MultiheadAttention(
            seq_length, self.num_heads, dropout=self.out_dropout
        )
        #  Output layer
        self.out_layer = nn.Linear(combined_dim, output_dim)

    def get_network(self, self_type="l", layers=-1):
        if self_type in ["l", "al", "vl"]:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ["a", "la", "va"]:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type == "l_mem":
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type == "a_mem":
            embed_dim, attn_dropout = self.d_a, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=self.num_heads,
            layers=max(self.layers, layers),
            attn_dropout=attn_dropout,
            relu_dropout=self.relu_dropout,
            res_dropout=self.res_dropout,
            embed_dropout=self.embed_dropout,
            attn_mask=self.attn_mask,
        )

    # def forward(self, x_l, x_a): #TODO FIXME THIS WAS ALREADY WORKING
    #     """
    #     text, and audio should have dimension [batch_size, seq_len, n_features]
    #     """

    #     x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
    #     x_a = x_a.transpose(1, 2)

    #     # Project the textual/audio features
    #     proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
    #     proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
    #     proj_x_l = proj_x_l.permute(2, 0, 1)
    #     proj_x_a = proj_x_a.permute(2, 0, 1)

    #     if self.lonly:
    #         # (V,A) --> L
    #         h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)    # Dimension (SEQ_LEN, BS, Dimensionality)
    #         h_ls = h_l_with_as
    #         h_ls = self.trans_l_mem(h_ls)
    #         if type(h_ls) == tuple:
    #             h_ls = h_ls[0]
    #         last_h_l = last_hs = h_ls#[-1]   # Take the last output for prediction

    #     if self.aonly:
    #         # (L,V) --> A
    #         h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
    #         h_as = h_a_with_ls
    #         h_as = self.trans_a_mem(h_as)
    #         if type(h_as) == tuple:
    #             h_as = h_as[0]
    #         last_h_a = last_hs = h_as#[-1]

    #     #last_hs = torch.cat([last_h_l, last_h_a], dim=1) #[BS,2*seq_lenght]

    #     #A residual block for language
    #     last_hs_proj_l = self.proj2(F.dropout(F.relu(self.proj1(last_h_l)), p=self.out_dropout, training=self.training))
    #     last_hs_proj_l += last_h_l
    #     #A residual block for audio
    #     last_hs_proj_a = self.proj4(F.dropout(F.relu(self.proj3(last_h_a)), p=self.out_dropout, training=self.training))
    #     last_hs_proj_a += last_h_a
    #     #output = self.out_layer(last_hs_proj)
    #     tgt2 = self.multihead_attn(last_hs_proj_l, last_hs_proj_a, last_hs_proj_a)[0]
    #     return tgt2.permute(1,0,2) #last_hs_proj, last_hs

    def forward(self, x_l, x_a):
        """
        text, and audio should have dimension [batch_size, seq_len, n_features]
        """
        x_l = F.dropout(
            x_l.transpose(1, 2), p=self.embed_dropout, training=self.training
        )
        x_a = x_a.transpose(1, 2)

        # Project the textual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_l = proj_x_l.permute(2, 0, 1)
        proj_x_a = proj_x_a.permute(2, 0, 1)

        if self.lonly:
            # (V,A) --> L
            h_l_with_as = self.trans_l_with_a(
                proj_x_l, proj_x_a, proj_x_a
            )  # Dimension (SEQ_LEN, BS, Dimensionality)
            h_ls = h_l_with_as
            h_ls = self.trans_l_mem(h_ls)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = last_hs = h_ls[-1]  # Take the last output for prediction

        if self.aonly:
            # (L,V) --> A
            h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
            h_as = h_a_with_ls
            h_as = self.trans_a_mem(h_as)
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = last_hs = h_as[-1]

        last_hs = torch.cat([last_h_l, last_h_a], dim=1)  # [BS,2*seq_lenght]

        # A residual block for language
        last_hs_proj_l = self.proj2(
            F.dropout(
                F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training
            )
        )
        last_hs_proj_l += last_hs

        # A residual block for audio
        # last_hs_proj_a = self.proj4(F.dropout(F.relu(self.proj3(last_h_a)), p=self.out_dropout, training=self.training))
        # last_hs_proj_a += last_h_a
        output = self.out_layer(last_hs_proj_l)
        # tgt2 = self.multihead_attn(last_hs_proj_l, last_hs_proj_a, last_hs_proj_a)[0]
        return output  # tgt2.permute(1,0,2) #last_hs_proj, last_hs
