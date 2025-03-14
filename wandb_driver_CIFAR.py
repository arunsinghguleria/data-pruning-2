'''
python3 wandb_driver_CIFAR.py --file_name exp5
'''

import argparse
import wandb
import time
from tqdm import tqdm
import pandas as pd


def run_wandb(args):
    df = pd.read_csv(f'/data/home1/arunsg/gitproject/data-pruning-2/results/cifar_experiment_results/{args.file_name}.csv')

    # wandb.init(project="pruning_CIFAR", group="tmp_group", config={'model':df.iloc[0]['prune_ratio']}, reinit=True)
    # wandb.init(project=f'pruning_CIFAR_{df.iloc[0]['sample']}', group="tmp_group", reinit=True)
    wandb.init(project=f'CIFAR10-LT-5', group="tmp_group", reinit=True)


    # dropout_str = "" if not args.dropout else "-dropout"


    # wandb.run.name = f"prune_after_epoch_5_90_50_20_nih_lt_{args.model}{dropout_str}-lr:{args.lr}-wd:{args.weight_decay}_" + wandb.run.name
    start = 0
    if('prune_ratio' in df.columns):
        wandb.run.name = args.file_name + '_' + df.iloc[start]['prune_ratio']


    for i in range(start,start+df.shape[0]):

        tmp = df.iloc[i].to_dict()
        epoch_stats = df.iloc[i].to_dict()
        epoch_num = tmp['epoch_num']
        
        if('sample' in epoch_stats):
            del epoch_stats['sample']
        
        if('prune_ratio' in epoch_stats):
            del epoch_stats['prune_ratio']
        
        # print(f'{epoch_num} - {epoch_stats}')
        wandb.log(epoch_stats, step=epoch_num)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, default=16, help='Batch size')
    args = parser.parse_args()

    run_wandb(args)
