import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task_name", type=str, default='_toxic_')

    # data loader and pre-process
    parser.add_argument("--data_path", type=str, default='./data')
    parser.add_argument("--save_path", type=str, default='./data')
    parser.add_argument("--preprocess", action='store_true')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--model_type", type=str, default='')
    parser.add_argument("--model_name_or_path", type=str, default='./data')

    # training & testing params
    parser.add_argument("--main_loss_weight", type=float, default=1.01)
    parser.add_argument("--decay", type=float, default=0.055)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--num_warmup_steps", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--print_iter", default=20, type=int)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--bsz", type=int, default=32)  # batch size
    parser.add_argument("--eval_bsz", type=int, default=1)
    parser.add_argument("--test_bsz", type=int, default=1)

    args = parser.parse_args(args=[])
    return args
