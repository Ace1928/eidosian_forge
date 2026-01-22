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
def dump_to_minify_after_dynamo(gm, args, compiler_name):
    subdir = os.path.join(minifier_dir(), 'checkpoints')
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)
    helper_for_dump_minify(generate_dynamo_fx_repro_string(gm, args, compiler_name, check_accuracy=config.repro_level == 4, save_dir=subdir, command='minify'))