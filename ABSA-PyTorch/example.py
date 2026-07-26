# -*- coding: utf-8 -*-
# file: infer.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2019. All Rights Reserved.

import torch
import numpy as np
import torch.nn.functional as F
import argparse

from ABSA.data_utils import build_tokenizer, build_embedding_matrix
from ABSA.models import LSTM, IAN, MemNet, ATAE_LSTM, AOA, TNet_LF ,TD_LSTM,RAM, MGAN


class Inferer:
    """A simple inference example"""

    def __init__(self, opt):
        self.opt = opt
        self.tokenizer = build_tokenizer(
            fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
            max_seq_len=opt.max_seq_len,
            dat_fname='{0}_tokenizer.dat'.format(opt.dataset))
        embedding_matrix = build_embedding_matrix(
            word2idx=self.tokenizer.word2idx,
            embed_dim=opt.embed_dim,
            dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset))
        self.model = opt.model_class(embedding_matrix, opt)
        #print('loading model {0} ...'.format(opt.model_name))
        self.model.load_state_dict(torch.load(opt.state_dict_path))
        self.model = self.model.to(opt.device)
        # switch model to evaluation mode
        self.model.eval()
        torch.autograd.set_grad_enabled(False)

    def evaluate(self, raw_texts, aspect):
        text_left, _, text_right = raw_texts.lower().strip().partition("$t$")
        # print(text_right)

        text_raw_indices = [self.tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)]
        text_raw_without_aspect_indices = [self.tokenizer.text_to_sequence(text_left + " " + text_right)]
        text_left_indices = [self.tokenizer.text_to_sequence(text_left)]
        text_left_with_aspect_indices = [self.tokenizer.text_to_sequence(text_left + " " + aspect)]
        text_right_indices = [self.tokenizer.text_to_sequence(text_right, reverse=True)]
        text_right_with_aspect_indices = [self.tokenizer.text_to_sequence(" " + aspect + " " + text_right, reverse=True)]
        aspect_indices = [self.tokenizer.text_to_sequence(aspect)]
        left_context_len = np.sum(text_left_indices != 0)
        aspect_len = np.sum(aspect_indices != 0)
        aspect_in_text = torch.tensor([left_context_len.item(), (left_context_len + aspect_len - 1).item()])
        context_indices = torch.tensor(text_raw_indices, dtype=torch.int64).to(self.opt.device)
        aspect_indices = torch.tensor(aspect_indices, dtype=torch.int64).to(self.opt.device)
        left= torch.tensor(text_left_with_aspect_indices,dtype=torch.int64).to(self.opt.device)
        right = torch.tensor(text_right_with_aspect_indices, dtype=torch.int64).to(self.opt.device)

        t_inputs = [context_indices, aspect_indices]
        t_outputs = self.model(t_inputs)

        t_probs = F.softmax(t_outputs, dim=-1).cpu().numpy()
        t_probs = int((t_probs.argmax(axis=-1) - 1))

        return t_probs


def ABSA(sentence, aspect):
    model_classes = {
        'atae_lstm': ATAE_LSTM,
        'ian': IAN,
        'memnet': MemNet,
        'aoa': AOA,
        'tnet_lf': TNet_LF,
        'td_lstm' : TD_LSTM,
        'ram' : RAM,
        'mgan' : MGAN
    }
    # set your trained models here
    model_state_dict_paths = {
        'atae_lstm': 'state_dict/atae_lstm_laptop_val_acc0.6583',
        'ian': 'state_dict/ian_laptop_val_acc0.5956',
        'memnet': 'state_dict/memnet_laptop_val_acc0.6991',
        'aoa': 'ABSA/state_dict/aoa_laptop_val_acc0.7716',
        'tnet_lf': 'state_dict/tnet_lf_laptop_val_acc0.7116',
        'td_lstm' : 'state_dict/td_lstm_laptop_val_acc0.6755',
        'mgan' : 'state_dict/mgan_laptop_val_acc0.6771'
    }

    class Option(object): pass

    opt = Option()
    opt.model_name = 'aoa'
    opt.model_class = model_classes[opt.model_name]
    opt.dataset = 'laptop'
    opt.dataset_file = {
        'train': 'ABSA/datasets/semeval14/Laptops_Train.xml.seg',
        'test': 'ABSA/datasets/semeval14/Laptops_Test_Gold.xml.seg'
    }
    opt.state_dict_path = model_state_dict_paths[opt.model_name]
    opt.embed_dim = 300
    opt.hidden_dim = 300
    opt.max_seq_len = 80
    opt.polarities_dim = 3
    opt.hops = 3
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    inf = Inferer(opt)
    final_sentence =sentence.lower().strip()
    t_probs = inf.evaluate(final_sentence,aspect)
    #print(t_probs)
    return t_probs


#ABSA("this thing has 2 sides , the material is great but $T$ is bad",'design')