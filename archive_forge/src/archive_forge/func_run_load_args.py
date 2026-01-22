import argparse
import copy
import functools
import logging
import os
import shutil
import sys
import textwrap
from importlib import import_module
from typing import Union
import torch
import torch.fx as fx
from torch._dynamo.debug_utils import (
from torch.fx.experimental.symbolic_shapes import fx_placeholder_targets
from torch.hub import tqdm
from .. import config
from ..backends.registry import lookup_backend, register_debug_backend
from ..debug_utils import clone_inputs_retaining_gradness
from math import inf
import torch
from torch import tensor, device
import torch.fx as fx
import torch._dynamo
from torch._dynamo.testing import rand_strided
from torch._dynamo.debug_utils import run_fwd_maybe_bwd
def run_load_args(options, mod, load_args):
    if not hasattr(load_args, '_version'):
        log.warning('load_args does not have a _version attribute, please file a bug to PyTorch and describe how you generate this repro script')
    elif load_args._version > 0:
        log.warning('load_args is version %s, but this version of PyTorch only supports version 0.  We will try to run it anyway but there may be an incompatibility; if so, try upgrading your version of PyTorch.', load_args._version)
    nop_reader = NopInputReader()
    load_args(nop_reader)
    with tqdm(desc='Loading inputs', total=nop_reader.total) as pbar:
        input_reader = InputReader(save_dir=options.save_dir, pbar=pbar)
        load_args(input_reader)
        args = input_reader.args
    return args