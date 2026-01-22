import argparse
import math
from abc import ABC
from functools import partial
import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from ..optimizer import AcceleratedOptimizer
from ..scheduler import AcceleratedScheduler
from .imports import is_megatron_lm_available, is_transformers_available
from .operations import recursively_apply, send_to_device
def model_provider_func(pre_process=True, post_process=True, add_encoder=True, add_decoder=True):
    """Build the model."""
    args = get_args()
    mode = 'pre-training' if args.pretraining_flag else 'fine-tuning'
    if args.rank == 0:
        print(f'Building {args.model_type_name} model in the {mode} mode.')
        print('The Megatron LM model weights are initialized at random in `accelerator.prepare`. Please use `accelerator.load_checkpoint` to load a pre-trained checkpoint matching the distributed setup.')
    if args.model_type_name == 'bert':
        if args.pretraining_flag:
            num_tokentypes = 2 if args.bert_binary_head else 0
            model = BertModel(num_tokentypes=num_tokentypes, add_binary_head=args.bert_binary_head, parallel_output=True, pre_process=pre_process, post_process=post_process)
        else:
            model = Classification(num_classes=args.num_labels, num_tokentypes=2, pre_process=pre_process, post_process=post_process)
    elif args.model_type_name == 'gpt':
        model = GPTModel(num_tokentypes=0, parallel_output=True, pre_process=pre_process, post_process=post_process)
    elif args.model_type_name == 't5':
        model = T5Model(num_tokentypes=0, parallel_output=True, pre_process=pre_process, post_process=post_process, add_encoder=add_encoder, add_decoder=add_decoder)
    else:
        raise ValueError(f'Unsupported model type: {args.model_type_name}')
    return model