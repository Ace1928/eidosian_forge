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
def repro_run(options, mod, load_args):
    opt_mod = torch._dynamo.optimize(options.backend)(mod)
    if options.accuracy != '':
        mod.eval()
        opt_mod.eval()
        with torch.cuda.amp.autocast(enabled=options.autocast):
            args = run_load_args(options, mod, load_args)
            assert same_two_models(mod, mod, args), 'Eager itself failed'
            if not same_two_models(mod, opt_mod, args):
                raise AccuracyError('Dynamo failed')
    else:
        with torch.cuda.amp.autocast(enabled=options.autocast):
            args = run_load_args(options, mod, load_args)
            ref = run_fwd_maybe_bwd(mod, args, only_fwd=options.only_fwd, disable_clone=True)
            del args
            args = run_load_args(options, mod, load_args)
            res = run_fwd_maybe_bwd(opt_mod, args, only_fwd=options.only_fwd, disable_clone=True)