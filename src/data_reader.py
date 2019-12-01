#!/usr/bin/env python
# coding: utf-8

# In[3]:

# import multiprocessing
# from functools import partial
import torch
from torch.utils.data import TensorDataset, DataLoader
# import torch.utils.data as data
# import transformers
import pandas as pd
import numpy as np

# from transformers import BertTokenizer

# from model_tools import add_split_column

# from src.model_tools import clip_to_max_len

AUX_TARGETS = [
    "target",
    "severe_toxicity",
    "obscene",
    "identity_attack",
    "insult",
    "threat", ]

IDENTITY_COLUMNS = [
    "male",
    "female",
    "homosexual_gay_or_lesbian",
    "christian",
    "jewish",
    "muslim",
    "black",
    "white",
    "psychiatric_or_mental_illness", ]

MAX_LEN = 300 - 1


# In[4]:


# tokenizer
def get_tokenizer(tokenizer_class, tokenizer_name):
    if (tokenizer_class is not None) and (tokenizer_name is not None):
        tokenizer = tokenizer_class.from_pretrained(tokenizer_name, do_lower_case=True)
    elif tokenizer_class and tokenizer_name:
        return None
    else:
        raise ValueError("Tokenizer information is missiing when class as {} "
                         "and name as {}".format(tokenizer_class, tokenizer_name))
    return tokenizer


# def my_tokenize(data, tokenizer_class, tokenizer_name):
#     tokenizer = get_tokenizer(tokenizer_class, tokenizer_name)
#     convert_line_uncased_with_t = partial(convert_line_uncased, tokenizer)
#     print("Tokenizing...")
#     with multiprocessing.Pool(processes=4) as pool:#进程池
#         text_list = data.comment_text.tolist()
#         sequences = pool.map(convert_line_uncased_with_t, text_list)
#     return sequences


# def convert_line_uncased(tokenizer, text):
#     tokens_a = tokenizer.tokenize(text)[:MAX_LEN]
#     one_token = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens_a)
#     one_token += [0] * (MAX_LEN - len(tokens_a))
#     return one_token

def my_tokenize(data, tokenizer_class, tokenizer_name):
    sequences_list = []
    tokenizer = get_tokenizer(tokenizer_class, tokenizer_name)

    for text_list in data.comment_text.tolist():
        tokens = tokenizer.tokenize(text_list)[:MAX_LEN]  # todo
        sequences = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens)
        sequences += [0] * (MAX_LEN - len(tokens))
        sequences_list.append(sequences)

    return sequences_list


def sequence_length(sequences):
    sequences = np.array(sequences)
    lengths = np.argmax(sequences == 0, axis=1)  # 每句话的实际词数 这里有例外：如果没有补零，默认值为0
    lengths[lengths == 0] = sequences.shape[1]  # 补零后每句话的词数 处理以上例外：为0的变成总长
    return lengths


def clip_to_max_len(batch):
    comment_id, x, y, lengths = map(torch.stack, zip(*batch))
    max_len = torch.max(lengths).item()
    return comment_id, x[:, :max_len], y


def clip_to_max_len_test(batch):
    comment_id, x, y, identity, lengths = map(torch.stack, zip(*batch))
    max_len = torch.max(lengths).item()
    return comment_id, x[:, :max_len], y, identity

# #如何取样本的
# def collate_fn(data):
#     def merge(sequences):
#         lengths = [len(seq) for seq in sequences]
#         padded_seqs = torch.zeros(len(sequences), max(lengths)).long()

#         for i, seq in enumerate(sequences):
#             end = lengths[i]
#             padded_seqs[i, :end] = seq[:end]
#         return padded_seqs, lengths

#     data.sort(key=lambda x: len(x[0]), reverse=True) ## sort by source seq
#     X, y, ind = zip(*data)

#     X, len_x = merge(X)

#     X = torch.LongTensor(X)

#     if(y[0] is None): pass
#     else: y = torch.LongTensor(y)

#     X = X.cuda()
#     if(y[0] is None): pass
#     else: y = y.cuda()

#     return X, y, ind
# def collate_fn(data):
#     def merge(sequences):
#         lengths = [len(seq) for seq in sequences]
#         padded_seqs = torch.zeros(len(sequences),
#                                   max(lengths)).long()  # pad with 0 (in pretrained GPT2 tokenizer: "!")
#         for i, seq in enumerate(sequences):
#             end = lengths[i]
#             padded_seqs[i, :end] = seq[:end]
#         return padded_seqs, lengths


def get_y(data):
    iden = data[IDENTITY_COLUMNS].fillna(0).values

    subgroup_target = np.hstack(  # 水平方向上堆叠
        [  # any全部F,则返回 F，如果有一个为 T，则返回 T
            (iden >= 0.5).any(axis=1, keepdims=True).astype(np.int),
            iden,
            iden.max(axis=1, keepdims=True),
        ])

    sub_target_weights = (~data[IDENTITY_COLUMNS].isna().values.any(axis=1, keepdims=True)).astype(np.int)

    weights = np.ones(len(data))
    weights += (iden >= 0.5).any(1)
    weights += (data["target"].values >= 0.5) & (iden < 0.5).any(1)
    weights += (data["target"].values < 0.5) & (iden >= 0.5).any(1)
    weights /= weights.mean()

    y_aux = data[AUX_TARGETS]

    y_torch = torch.tensor(
        np.hstack([data.target.values[:, None],
                   weights[:, None],
                   y_aux,
                   subgroup_target,
                   sub_target_weights])).float()
    return y_torch


# def get_dataloader(data, args, tokenizer_class, tokenizer_name):

#     data = pd.read_csv(os.path.join(args.data_path, "norm_data.csv"))
#     data = add_split_column(data)
def get_dataloader(args, tokenizer_class, tokenizer_name):
    # train_data = data.query('split == "train"')
    # valid_data = data.query('split == "valid"')
    # test_data = data.query('split == "test"')

    train_data = pd.read_csv('data/debiased_data_zong300.csv', sep='\t', engine='python')
    train_data.head()
    test_data = pd.read_csv('data/test_public_expanded.csv')  # , sep='\t', engine='python')
    test_data.head()

    y_train_torch = get_y(train_data)
    train_sequences = my_tokenize(train_data, tokenizer_class, tokenizer_name)
    train_lengths = sequence_length(train_sequences)
    train_dataset = TensorDataset(torch.tensor(train_data.id), torch.LongTensor(train_sequences), y_train_torch,
                                  torch.from_numpy(train_lengths))

    # y_valid_torch = get_y(valid_data) valid_sequences = my_tokenize(valid_data, tokenizer_class, tokenizer_name)
    # valid_lengths = sequence_length(valid_sequences) valid_dataset = torch.utils.data..TensorDataset(
    # torch.from_numpy(valid_sequences).long(), y_valid_torch, torch.from_numpy(valid_lengths))

    y_test_torch = torch.tensor(test_data.toxicity.values[:, None]).float()
    identities_torch = torch.tensor(test_data[IDENTITY_COLUMNS].fillna(0).values[:, None]).float()
    test_sequences = my_tokenize(test_data, tokenizer_class, tokenizer_name)
    test_lengths = sequence_length(test_sequences)
    tt = torch.LongTensor(test_sequences)
    # print(tt.shape)
    # print(y_test_torch.shape)
    # print(test_lengths.shape)

    test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_data.id), tt, y_test_torch,
                                                  identities_torch, torch.from_numpy(test_lengths))

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.bsz,
                              shuffle=True,
                              collate_fn=clip_to_max_len)
    #     valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
    #                                             batch_size=1, #args.test_bsz,
    #                                             shuffle=False,
    #                                             collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.test_bsz,
                                              shuffle=False,
                                              collate_fn=clip_to_max_len_test)
    dataloaders = {
        'train': train_loader,
        'test': test_loader
    }

    return dataloaders
#     return train_loader, valid_loader, test_loader


# In[6]:


# data = pd.read_csv('../data/mini_test.csv', sep='\t', engine='python')


# In[ ]:
