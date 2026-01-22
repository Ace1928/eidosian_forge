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
def repro_minify(options, mod, load_args):
    args = run_load_args(options, mod, load_args)
    if not options.accuracy:
        compiler_fn = lookup_backend('dynamo_minifier_backend')
    else:
        compiler_fn = lookup_backend('dynamo_accuracy_minifier_backend')
    if options.backend is None:
        raise RuntimeError('Compiler name is None - this likely means that a custom compiler was called by torchdynamo. Please remove this error, import your custom compiler function, and replace the backend=None line in run_repro to backend=<my_imported_custom_function>')
    dynamo_minifier_backend = functools.partial(compiler_fn, compiler_name=options.backend)
    opt_mod = torch._dynamo.optimize(dynamo_minifier_backend)(mod)
    with torch.cuda.amp.autocast(enabled=options.autocast):
        opt_mod(*args)