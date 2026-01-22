import copy
import functools
import getpass
import itertools
import logging
import os
import subprocess
import tempfile
import textwrap
from collections import Counter
from importlib import import_module
from typing import Callable, Optional, TypeVar
import torch
import torch._prims_common as utils
import torch._subclasses.meta_utils
from torch._dynamo.testing import rand_strided
from torch._prims_common import is_float_dtype
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._content_store import ContentStoreReader, ContentStoreWriter
from . import config
from .utils import clone_inputs, get_debug_dir
import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
def backend_accuracy_fails(gm, example_inputs, compiler_fn, only_fwd=False, *, require_fp64=False, ignore_non_fp=False):
    try:
        compiled_gm = compiler_fn(copy.deepcopy(gm), clone_inputs_retaining_gradness(example_inputs))
        return not same_two_models(gm, compiled_gm, example_inputs, only_fwd, require_fp64=require_fp64, ignore_non_fp=ignore_non_fp)
    except Exception as e:
        log.exception('While minifying the program in accuracy minification mode, ran into a runtime exception which is likely an unrelated issue. Skipping this graph')
        return False