#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import io
import os
# import re

import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
# import scipy.stats as stats  # 统计函数库
import seaborn as sns  # 可视化
from six.moves import range
from six.moves import zip
# from sklearn import metrics
import torch
from torch import nn


# In[ ]:


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# In[ ]:


### Plotting.
def per_subgroup_scatterplots(df,
                              subgroup_col,
                              values_col,
                              title='',
                              y_lim=(0.8, 1.0),
                              figsize=(15, 5),
                              point_size=8,
                              file_name='plot'):
    """Displays a series of one-dimensional scatterplots, 1 scatterplot per subgroup.
    Args:
          df: DataFrame contain subgroup_col and values_col.
          subgroup_col: Column containing subgroups.
          values_col: Column containing collection of values to plot (each cell
            should contain a sequence of values, e.g. the AUCs for multiple models
            from the same family).
          title: Plot title.
          y_lim: Plot bounds for y axis.
          figsize: Plot figure size.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    for i, (_, row) in enumerate(df.iterrows()):
    # For each subgroup, we plot a 1D scatterplot. The x-value is the position
    # of the item in the dataframe. To change the ordering of the subgroups,
    # sort the dataframe before passing to this function.
        x = [i] * len(row[values_col])
        y = row[values_col]
        ax.scatter(x, y, s=point_size)
        
    ax.set_xticklabels(df[subgroup_col], rotation=90)
    ax.set_xticks(list(range(len(df))))
    ax.set_ylim(y_lim)
    ax.set_title(title)
    fig.tight_layout()
#     fig.savefig('/tmp/%s_%s.eps' % (file_name, values_col), format='eps')


def plot_metric_heatmap(bias_metrics_results,
                        models,
                        metrics_list,
                        cmap=None,
                        show_subgroups=True,
                        vmin=0,
                        vmax=1.0):
    df = bias_metrics_results.set_index(SUBGROUP)
    columns = []
    # Add vertical lines around all columns.
    vlines = [i * len(models) for i in range(len(metrics_list) + 1)]
    for metric in metrics_list:
        for model in models:
              columns.append(column_name(model, metric))
    num_rows = len(df)
    num_columns = len(columns)
    fig = plt.figure(figsize=(num_columns, 0.5 * num_rows))
    ax = sns.heatmap(
          df[columns],
          annot=True,
          fmt='.2',
          cbar=False,
          cmap=cmap,
          vmin=vmin,
          vmax=vmax)
    ax.xaxis.tick_top()
    if not show_subgroups:
        ax.yaxis.set_visible(False)
    ax.yaxis.set_label_text('')
    plt.xticks(rotation=90)
    ax.vlines(vlines, *ax.get_ylim())

    return ax


def plot_auc_heatmap(bias_metrics_results, models, color_palette=None):
    if not color_palette:
        # Hack to align these colors with the AEG colors below.
        cmap = sns.color_palette('coolwarm', 9)[4:]
        cmap.reverse()
    else:
        cmap = color_palette
    return plot_metric_heatmap(
      bias_metrics_results, models, AUCS, cmap=cmap, show_subgroups=True, vmin=0.5, vmax=1.0)


# In[ ]:


def add_split_column(df, train_frac=0.6, test_frac=0.2):
    """Adds a 'split' column to the data frame, assigning "train", "dev", or "test" to each row randomly."""
    assert(train_frac + test_frac <= 1.0)
    train_num = int(round(len(df) * train_frac))
    test_num = int(round(len(df) * test_frac)) 
    print(train_num, test_num)
    
    train_items = df.index.isin(df.sample(train_num).index)
    df.loc[train_items, 'split'] = 'train'

    test_items = df.index.isin(df[df['split'].isnull()].sample(test_num).index)
    df.loc[test_items, 'split'] = 'test'

    dev_items = df[df['split'].isnull()].index
    df.loc[dev_items, 'split'] = 'dev'
    return df


def filter_frame(frame, keyword=None, length=None):
    """Filters DataFrame to comments that contain the keyword as a substring and fit within length."""
    if keyword:
        # frame = frame[frame['comment_text'].str.contains(keyword, case=False)]
        frame = frame[frame[keyword] >= 0.5]
    if length:
        frame = frame[frame['length'] <= length]
    return frame


def should_decay(name):
    return not any(n in name for n in ("bias", "LayerNorm.bias", "LayerNorm.weight"))


def prepare_loss(main_loss_weight):
    # loss function from https://github.com/iezepov/combat-wombat-bias-in-toxicity.git
    def custom_loss(data, targets):
        bce_loss_1 = nn.BCEWithLogitsLoss(targets[:, 1:2])(data[:, :1], targets[:, :1])
        bce_loss_2 = nn.BCEWithLogitsLoss()(data[:, 1:7], targets[:, 2:8])
        bce_loss_3 = nn.BCEWithLogitsLoss(targets[:, 19:20])(data[:, 7:18], targets[:, 8:19])
        return main_loss_weight * bce_loss_1 + bce_loss_2 + bce_loss_3 / 4

    return custom_loss
