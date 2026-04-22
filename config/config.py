import os
import argparse
import json
import shutil
from .file_utils import ensure_dirs
from .macro import *

class Config(object):
    def __init__(self, phase):
        self.is_train = phase == "train"

        # init hyperparameters and parse from command-line
        parser, args = self.parse()
        self.set_configuration()

        # set as attributes
        print("----Experiment Configuration-----")
        for k, v in args.__dict__.items():
            print("{0:20}".format(k), v)
            self.__setattr__(k, v)

        # experiment paths
        self.exp_dir = os.path.join(self.proj_dir, self.exp_name)
        if phase == "train" and args.cont is not True and os.path.exists(self.exp_dir):
            response = input('Experiment log/model already exists, overwrite? (y/n) ')
            if response != 'y':
                exit()
            shutil.rmtree(self.exp_dir)

        self.log_dir = os.path.join(self.exp_dir, 'log')
        self.model_dir = os.path.join(self.exp_dir, 'model')
        ensure_dirs([self.log_dir, self.model_dir])

        # GPU usage
        if args.gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)

        # save this configuration
        if self.is_train:
            with open('{}/config.txt'.format(self.exp_dir), 'w') as f:
                json.dump(args.__dict__, f, indent=2)

    def set_configuration(self):
        self.args_dim = ARGS_DIM # 256
        self.cad_n_args = CAD_N_ARGS  #16
        self.svg_n_args = SVG_N_ARGS  #8
        self.cad_n_commands = len(CAD_COMMANDS)  #6: 0Line, 1Arc, 2Circle, 3EOS, 4SOL, 5Ext
        self.svg_n_commands = len(SVG_COMMANDS)  #4: SOS, EOS, L, C

        self.n_layers = 4                # Number of Encoder blocks
        self.n_layers_decode = 4         # Number of Decoder blocks
        self.n_heads = 8                 # Transformer config: number of heads
        self.dim_feedforward = 512       # Transformer config: FF dimensionality
        self.d_model = 256               # Transformer config: model dimensionality
        self.dropout = 0.1               # Dropout rate used in basic layers and Transformers
        self.dim_z = 256                 # Latent vector dimensionality

        self.text_model="bert_large_uncased"
        self.text_max_len = 512
        self.text_cache_dir = "/home/ubuntu/.cache/huggingface/hub/models--google-bert--bert-large-uncased/snapshots/6da4b6a26a1877e173fca3225479512db81a5e5b"
        self.text_embed_dim = 1024
        self.adapt_dim = self.d_model
        self.use_text = True
        self.adaptive_layer_type = "linear"


        self.cad_max_n_ext = CAD_MAX_N_EXT #10
        self.cad_max_n_loops = CAD_MAX_N_LOOPS #6
        self.cad_max_n_curves = CAD_MAX_N_CURVES #15

        self.cad_max_total_len = CAD_MAX_TOTAL_LEN #60
        self.svg_max_total_len = SVG_MAX_TOTAL_LEN #100

        self.loss_weights = {
            "loss_cmd_weight": 1.0,
            "loss_args_weight": 2.0,
        }

        
    def parse(self):
        """initiaize argument parser. Define default hyperparameters and collect from command-line arguments."""
        parser = argparse.ArgumentParser()

        parser.add_argument('--proj_dir', type=str, default="proj_log", help="path to project folder where models and logs will be saved")
        parser.add_argument('--data_root', type=str, default="data", help="path to source data folder")
        parser.add_argument('--exp_name', type=str, default=os.getcwd().split('/')[-1], help="name of this experiment")
        parser.add_argument('-g', '--gpu_ids', type=str, default='0', help="gpu to use, e.g. 0  0,1,2. CPU not supported.")        
        
        parser.add_argument('--batch_size', type=int, default=256, help="batch size")
        parser.add_argument('--num_workers', type=int, default=8, help="number of workers for data loading")  #用于并行加载数据的子进程数

        parser.add_argument('--nr_epochs', type=int, default=100, help="total number of epochs to train")
        parser.add_argument('--lr', type=float, default=1e-3, help="initial learning rate")
        parser.add_argument('--grad_clip', type=float, default=1.0, help="initial learning rate")
        parser.add_argument('--warmup_step', type=int, default=2000, help="step size for learning rate warm up")
        parser.add_argument('--continue', dest='cont',  action='store_true', help="continue training from checkpoint")  #False表示从头开始训练  --continue
        parser.add_argument('--ckpt', type=str, default='latest', required=False, help="desired checkpoint to restore")
        parser.add_argument('--vis', action='store_true', default=False, help="visualize output in training")
        parser.add_argument('--save_frequency', type=int, default=50, help="save models every x epochs")
        parser.add_argument('--val_frequency', type=int, default=10, help="run validation every x iterations")
        parser.add_argument('--vis_frequency', type=int, default=2000, help="visualize output every x iterations")
        
        args = parser.parse_args()
        return parser, args