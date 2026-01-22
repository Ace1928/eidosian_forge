from __future__ import annotations
import contextlib
import dataclasses
import functools
import logging
import os
import queue
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from ctypes import byref, c_size_t, c_void_p
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue
from typing import (
import torch
from torch import multiprocessing
from torch._dynamo.testing import rand_strided
from torch._inductor import ir
from torch._inductor.codecache import CUDACodeCache, DLLWrapper, PyCodeCache
from . import config
from .utils import do_bench
from .virtualized import V
@classmethod
def from_irnodes(cls, irnodes: Union[LayoutOrBuffer, Sequence[LayoutOrBuffer]]) -> Union[TensorMeta, List[TensorMeta]]:
    if isinstance(irnodes, Sequence):
        result: List[Any] = [cls.from_irnodes(x) for x in irnodes]
        assert all((isinstance(x, TensorMeta) for x in result))
        return result
    node = irnodes
    if isinstance(node, ir.Layout):
        node = ir.Buffer('fake', node)
    dtype = node.get_dtype()
    assert dtype is not None
    return TensorMeta(device=node.get_device(), dtype=dtype, sizes=V.graph.sizevars.size_hints(node.get_size(), fallback=config.unbacked_symint_fallback), strides=V.graph.sizevars.size_hints(node.get_stride(), fallback=config.unbacked_symint_fallback), offset=V.graph.sizevars.size_hint(node.get_layout().offset, fallback=config.unbacked_symint_fallback))