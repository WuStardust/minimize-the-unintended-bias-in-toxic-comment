# import os
# import pickle
# import torch
# from torch import nn
# import torch.nn.functional as F
# from transformers import (BertConfig, BertForSequenceClassification, BertTokenizer,
#                             XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer,
#                             RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
#                             DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer)

from transformers import BertForSequenceClassification


class ToxicClassifier(BertForSequenceClassification):
    def __init__(self, config):
        super(ToxicClassifier, self).__init__(config)
        self.bert = BertForSequenceClassification(config)

        # self.custom_loss = prepare_loss(1.01)
        # self.bert = BertForSequenceClassification.from_pretrained(pretrained_type, num_labels=18).to(device=device)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, labels=labels)
        # logits = outputs[1:18]
        # loss = self.custom_loss(logits, labels)
        # outputs = (loss,) + outputs[1:]
        return outputs[0]
