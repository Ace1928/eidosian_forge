import builtins
import functools
import inspect
import itertools
import logging
import sys
import textwrap
import time
from io import StringIO
from typing import Any, Callable, Dict, List, Optional, Type, Union
from unittest.mock import patch
import sympy
import torch
from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import counters, identity, preserve_rng_state
from . import config, ir
from .autotune_process import TensorMeta, TritonBenchmarkRequest
from .codecache import code_hash, PersistentCache, PyCodeCache
from .codegen.common import ChoiceCaller, IndentedBuffer, KernelTemplate
from .codegen.triton import texpr, TritonKernel, TritonPrinter, TritonScheduling
from .codegen.triton_utils import config_of, signature_to_meta
from .exc import CUDACompileError
from .utils import do_bench, Placeholder, sympy_dot, sympy_product, unique
from .virtualized import V
from . import lowering  # noqa: F401
def make_load(self, name, indices, mask):
    """
        Optional helper called from template code to generate the code
        needed to load from an tensor.
        """
    assert isinstance(indices, (list, tuple))
    assert isinstance(name, str)
    assert isinstance(mask, str)
    stride = self.named_input_nodes[name].get_stride()
    indices = list(map(TritonPrinter.paren, indices))
    assert len(indices) == len(stride)
    index = ' + '.join((f'{texpr(self.rename_indexing(s))} * {i}' for s, i in zip(stride, indices)))
    return f'tl.load({name} + ({index}), {mask})'