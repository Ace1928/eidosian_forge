import argparse
import json
import os
import re
import sys
import types
import torch
from transformers import AutoTokenizer, GPT2Config
from transformers.modeling_utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME, shard_checkpoint
def get_megatron_sharded_states(args, tp_size, pp_size, pp_rank):
    """
    Get sharded checkpoints from NVIDIA Megatron-LM checkpoint based on the provided tensor parallel size, pipeline
    parallel size and pipeline parallel rank.

    Args:
        args (argparse.Namespace): the arguments to the script
        tp_size (int): the tensor parallel size
        pp_size (int): the pipeline parallel size
        pp_rank (int): the pipeline parallel rank
    """
    tp_state_dicts = []
    for i in range(tp_size):
        sub_dir_name = f'mp_rank_{i:02d}' if pp_size == 1 else f'mp_rank_{i:02d}_{pp_rank:03d}'
        for checkpoint_name in ['model_optim_rng.pt', 'model_rng.pt']:
            checkpoint_path = os.path.join(args.load_path, sub_dir_name, checkpoint_name)
            if os.path.isfile(checkpoint_path):
                break
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        tp_state_dicts.append(state_dict)
    return tp_state_dicts