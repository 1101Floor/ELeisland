"""
run.py — MambaTS + Gumbel-Sinkhorn 入口
用法示例:
  python run.py --task_name long_term_forecast --is_training 1 \
    --model_id test --model MambaTS --data ETTm1 \
    --root_path ./data/ETT/ --data_path ETTm1.csv \
    --features M --seq_len 96 --pred_len 96 \
    --patch_len 16 --stride 8 --enc_in 7 \
    --d_model 512 --e_layers 2 --dropout 0.1 \
    --learning_rate 0.0001 --train_epochs 10 --batch_size 32 \
    --gs_d_hidden 128 --gs_sinkhorn_iters 20 --gs_temperature 1.0
"""
import argparse
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.print_args import print_args
from utils.tools import set_seed

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MambaTS + Gumbel-Sinkhorn')

    # ---- basic config ----
    parser.add_argument('--task_name', type=str,
                        default='long_term_forecast',
                        help='task name: [long_term_forecast, short_term_forecast]')
    parser.add_argument('--is_training', type=int,  default=1,
                        help='status: 1=train+test, 0=test only')
    parser.add_argument('--model_id', type=str, default='test',
                        help='model id')
    parser.add_argument('--seed', type=int, default=3047, help='random seed')
    parser.add_argument('--model', type=str, default='MambaTS',
                        help='model name')

    # ---- data loader ----
    parser.add_argument('--data', type=str, default='custom',
                        help='dataset type')
    parser.add_argument('--root_path', type=str, default='./mine_results/',
                        help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='selected_features_data.csv',
                        help='data file')
    parser.add_argument('--features', type=str, default='MS',
                        help='[M, S, MS]: multivariate/univariate/multi-to-uni')
    parser.add_argument('--target', type=str, default='ME_FO_consumption',
                        help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='s',
                        help='freq for time features encoding')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/',
                        help='location of model checkpoints')
    parser.add_argument('--visualization', type=str, default='./test_results',
                        help='location of visualization results')
    parser.add_argument('--results', type=str, default='./results',
                        help='location of results')

    # ---- forecasting task ----
    parser.add_argument('--seq_len', type=int, default=96,
                        help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48,
                        help='start token length')
    parser.add_argument('--pred_len', type=int, default=96,
                        help='prediction sequence length')
    parser.add_argument('--max_pred_len', type=int, default=-1,
                        help='max prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly',
                        help='subset for M4')
    parser.add_argument('--inverse', action='store_true', default=False,
                        help='inverse output data')

    # ---- model define ----
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=69,
                        help='encoder input size (= number of variables)')
    parser.add_argument('--dec_in', type=int, default=69,
                        help='decoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--d_model', type=int, default=512,
                        help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='num of heads')
    parser.add_argument('--e_layers', type=int, default=3,
                        help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1,
                        help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048,
                        help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25,
                        help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false', default=True,
                        help='whether to use distilling in encoder')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding')
    parser.add_argument('--activation', type=str, default='gelu',
                        help='activation')
    parser.add_argument('--output_attention', action='store_true',
                        help='whether to output attention')

    # ---- PatchTST ----
    parser.add_argument('--patch_len', type=int, default=16,
                        help='patch length for tokenization')
    parser.add_argument('--stride', type=int, default=8,
                        help='stride for patching')

    # ---- MambaTS ----
    parser.add_argument('--use_casual_conv', action='store_true', default=True,
                        help='use causal conv in Mamba blocks')

    # ★ Gumbel-Sinkhorn 参数（替代原 VPT_mode / ATSP_solver）
    parser.add_argument('--gs_d_hidden', type=int, default=32,
                        help='hidden dim of Gumbel-Sinkhorn ordering network')
    parser.add_argument('--gs_sinkhorn_iters', type=int, default=10,
                        help='number of Sinkhorn iterations')
    parser.add_argument('--gs_temperature', type=float, default=2,
                        help='initial temperature for Sinkhorn')
    parser.add_argument('--gs_hard_at_train', action='store_true', default=False,
                        help='use hard permutation during training (for debug)')

    # ---- optimization ----
    parser.add_argument('--num_workers', type=int, default=0,
                        help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1,
                        help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=20,
                        help='train epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size')
    parser.add_argument('--patience', type=int, default=5,
                        help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--des', type=str, default='test',
                        help='exp description')
    parser.add_argument('--loss', type=str, default='MSE',
                        help='loss function: [MSE, L1]')
    parser.add_argument('--lradj', type=str, default='type1',
                        help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', default=False,
                        help='use automatic mixed precision')

    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
    parser.add_argument('--no_lradj', action='store_true', default=False)
    parser.add_argument('--reg_weight', default=1e-5, type=float,
                        help='regularization weight')
    parser.add_argument('--momentum', default=0.99, type=float, help='momentum')
    parser.add_argument('--lradj_by_iter', action='store_true', default=False)
    parser.add_argument('--warmup_steps', default=0.1, type=float, help='warmup')
    parser.add_argument('--iters_per_epoch', default=None, type=str,
                        help='warmup')

    # ---- GPU ----
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False,
                        help='use multiple gpus')
    parser.add_argument('--devices', type=str, default='0,1,2,3',
                        help='device ids of multiple gpus')

    # ---- de-stationary projector params ----
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector')
    parser.add_argument('--p_hidden_layers', type=int, default=2,
                        help='number of hidden layers in projector')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.iters_per_epoch is not None:
        args.iters_per_epoch = eval(args.iters_per_epoch)

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    set_seed(args.seed)

    print('Args in experiment:')
    print_args(args)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            setting = (
                f'{args.task_name}_{args.model_id}_{args.model}_{args.data}_'
                f'ft{args.features}_sl{args.seq_len}_ll{args.label_len}_'
                f'pl{args.pred_len}_dm{args.d_model}_nh{args.n_heads}_'
                f'el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}_'
                f'patch{args.patch_len}_stride{args.stride}_'
                f'gs{args.gs_d_hidden}_temp{args.gs_temperature}_{args.des}_{ii}'
            )

            exp = Exp(args)
            print(f'>>>>>>>start training: {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
            exp.train(setting)

            print(f'>>>>>>>testing: {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            set_seed(args.seed)
            exp.test(setting)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = (
            f'{args.task_name}_{args.model_id}_{args.model}_{args.data}_'
            f'ft{args.features}_sl{args.seq_len}_ll{args.label_len}_'
            f'pl{args.pred_len}_dm{args.d_model}_nh{args.n_heads}_'
            f'el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}_'
            f'patch{args.patch_len}_stride{args.stride}_'
            f'gs{args.gs_d_hidden}_temp{args.gs_temperature}_{args.des}_{ii}'
        )

        exp = Exp(args)
        print(f'>>>>>>>testing: {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        set_seed(args.seed)
        exp.test(setting, test=1)

        torch.cuda.empty_cache()
