import itertools
from typing import List, Tuple
import torch
import triton
import triton.language as tl
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time
def our_early_config_prune(config, named_args):
    new_named_args = named_args.copy()
    new_named_args['M'] = named_args['M1'] + named_args['M2'] + named_args['M3']
    new_named_args['N'] = named_args['N1'] + named_args['N2'] + named_args['N3']
    new_named_args['K'] = named_args['K1'] + named_args['K2'] + named_args['K3']
    new_named_args['A'] = named_args['A11']
    new_named_args['B'] = named_args['B11']
    new_named_args['C'] = named_args['C11']
    return early_config_prune(config, new_named_args)