from config import get_args
import pandas as pd
import os
from src.data_reader import get_dataloader
# from transformers import (WEIGHTS_NAME, AdamW, WarmupLinearSchedule,
#                             BertConfig, BertForSequenceClassification, BertTokenizer,
#                             XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer,
#                             RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
#                             DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer)
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer

from src.model import ToxicClassifier

import torch
from torch import nn

# import copy

import datetime

# from collections import OrderedDict

from src.model_tools import should_decay, prepare_loss

datetimestr = datetime.datetime.now().strftime('%Y-%b-%d_%H_%M_%S')

ACCUM_STEPS = 2


# MODEL_CLASSES = {
#     'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
#     'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
#     'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
#     'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer)
# }


def train_model(args, model, epochs, optimizer, scheduler, dataloaders, device, print_iter, patience):
    # When patience_counter > patience, the training will stop
    patience_counter = 0
    # Statistics to record
    best_train_acc = 0
    best_val_acc = 0
    train_accs = []
    val_accs = []
    train_losses = []
    val_losses = []
    # best_model_weights = None

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    for epoch in range(epochs):
        print('\n Epoch {}'.format(epoch), flush=True)
        print('=' * 20, flush=True)

        for phase in ['train']:  # , 'val']:
            print('Phase [{}]'.format(phase), flush=True)
            print('-' * 10, flush=True)

            if phase == 'train':
                model.train()  # pretrained_type to training mode
            else:
                model.eval()  # Set model to evaluate mode

            dataloader = dataloaders[phase]
            this_acc = 0
            this_loss = 0
            iter_per_epoch = 0

            custom_loss = prepare_loss(args.main_loss_weight)

            for iteration, (_, input_data, labels) in enumerate(dataloader):
                iter_per_epoch += 1

                input_data = input_data.to(device=device)
                labels = labels.to(device=device)

                # optimizer.zero_grad()
                model.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    labels_pre = model(input_data, attention_mask=(input_data > 0))
                    loss = custom_loss(labels_pre, labels)
                    # acc = torch.sum(logits.argmax(dim=1) == labels).item() / logits.size(dim=0)
                    this_loss += loss.item()
                    # this_acc += acc

                    if phase == 'train':
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                        optimizer.step()
                        scheduler.step()
                        if iteration % print_iter == 0:
                            print('Iteration {}: loss = {:4f}'.format(iteration, loss), flush=True)

                this_loss = (this_loss / iter_per_epoch)
                # this_acc = (this_acc / iter_per_epoch)
                # print('Loss = {:4f}, Acc = {:4f}'.format(this_loss, this_acc), flush=True)

                if phase == 'train':
                    train_losses.append(this_loss)
                    train_accs.append(this_acc)
                    if this_acc > best_train_acc:
                        best_train_acc = this_acc
                else:
                    patience_counter += 1
                    val_losses.append(this_loss)
                    val_accs.append(this_acc)
                    if this_acc > best_val_acc:
                        best_val_acc = this_acc
                        patience_counter = 0
                        # best_model_weights = copy.deepcopy(model.state_dict())
                        # best_model_weights = {k: v.to('cpu') for k, v in best_model_weights.items()}
                        # best_model_weights = OrderedDict(best_model_weights)
                    elif patience_counter == patience:
                        print('Stop training because running out of patience!', flush=True)
                        # save((
                        #     train_losses,
                        #     val_losses,
                        #     best_train_acc,
                        #     best_val_acc,
                        #     train_accs,
                        #     val_accs
                        # ), os.path.join(args.save_path, 'results' + task_name + datetimestr + '.pt'))
                        # save(best_model_weights,
                        #      os.path.join(args.save_path, 'best_model_weights' + task_name + datetimestr + '.pt'))
                        exit(1)

    # torch.save(model.state_dict(), args.save_path + '/model_weights' + task_name + datetimestr + '.pt')
    model.save_pretrained(args.save_path)
    # save(best_model_weights, os.path.join(args.save_path, 'best_model_weights' + task_name + datetimestr + '.pt'))
    # save((
    #     train_losses,
    #     val_losses,
    #     best_train_acc,
    #     best_val_acc,
    #     train_accs,
    #     val_accs
    # ), os.path.join('./data', 'results' + task_name + datetimestr + '.pt'))


def test_model(model, dataloader, device, print_iter):
    results = []
    model.eval()
    for iteration, (comment_id, input_data, labels, identity) in enumerate(dataloader):
        input_data = input_data.to(device=device)
        labels_pre = model(input_data, attention_mask=(input_data > 0)).cpu()

        results.append({
            'id': comment_id[0].numpy(),
            'target': labels[0][0].numpy(),
            'prediction_score': labels_pre[0][0].detach().numpy(),
            'male': identity[0][0][0].numpy(),
            'female': identity[0][0][1].numpy(),
            'homosexual_gay_or_lesbian': identity[0][0][2].numpy(),
            'christian': identity[0][0][3].numpy(),
            'jewish': identity[0][0][4].numpy(),
            'muslim': identity[0][0][5].numpy(),
            'black': identity[0][0][6].numpy(),
            'white': identity[0][0][7].numpy(),
            'psychiatric_or_mental_illness': identity[0][0][8].numpy()
        })
        if iteration % print_iter == 0:
            print('Iteration {}: '.format(iteration), flush=True)

    pd.DataFrame(results).to_csv('./data/results.csv', sep='\t', encoding='utf-8', index=False)


def main():
    print('Program Start...')
    args = get_args()

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    # config_class = BertConfig
    # model_class = BertForSequenceClassification
    # tokenizer_class = BertTokenizer

    # config = config_class.from_pretrained(args['model_name_or_path'])  # eg: bert-base-uncased

    print('get data loader...')
    dataloaders = get_dataloader(args, BertTokenizer, 'bert-base-uncased')  # args.model_name_or_path)

    # model = model_class.from_pretrained(args['model_name_or_path'], from_tf=bool('.ckpt' in args.model_name_or_path),
    #                                     config=config)
    model = ToxicClassifier.from_pretrained('bert-base-uncased', num_labels=18).to(device=device)

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if should_decay(n)],
            "weight_decay": args.decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if not should_decay(n)],
            "weight_decay": 0.00,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr,
                      correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    # PyTorch scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps,
                                                num_training_steps=args.epochs * len(dataloaders['train']) // ACCUM_STEPS)

    print('start training...')
    train_model(
        args=args,
        model=model,
        epochs=args.epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        print_iter=args.print_iter,
        patience=args.patience,
        dataloaders=dataloaders
    )

    print('Start test...')
    model = ToxicClassifier.from_pretrained(args.model_name_or_path, num_labels=18).to(device=device)

    test_model(
        model=model,
        device=device,
        print_iter=args.print_iter,
        dataloader=dataloaders['test']
    )


if __name__ == "__main__":
    main()
