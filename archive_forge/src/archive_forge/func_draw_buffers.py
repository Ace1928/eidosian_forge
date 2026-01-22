import collections
import contextlib
import cProfile
import dataclasses
import functools
import itertools
import logging
import os
import os.path
import pickle
import pstats
import shutil
import subprocess
from typing import Any, Dict, List, Optional
from unittest.mock import patch
from functorch.compile import draw_graph, get_aot_graph_name, get_graph_being_compiled
import torch
from torch import fx as fx
from torch._dynamo.repro.after_aot import save_graph_repro, wrap_compiler_debug
from torch._dynamo.utils import get_debug_dir
from torch.fx.graph_module import GraphModule
from torch.fx.passes.shape_prop import _extract_tensor_metadata, TensorMetadata
from torch.fx.passes.tools_common import legalize_graph
from torch.utils._pytree import tree_map
from . import config, ir  # noqa: F811, this is needed
from .scheduler import (
from .virtualized import V
from torch._inductor.debug import load_args_and_run_compile_fx_inner
def draw_buffers(nodes: List[BaseSchedulerNode], print_graph=False, fname=None):
    """
    Draw a graph in fname.svg.
    """
    if not has_dot():
        log.warning('draw_buffers() requires `graphviz` package')
        return
    if fname is None:
        fname = get_graph_being_compiled()
    graph = create_fx_from_snodes(nodes)
    for node in graph.nodes:
        if 'fusion_meta' not in node.meta:
            continue
        group = node.meta['fusion_meta'].group
        if isinstance(group, tuple):
            if isinstance(group[1], int):
                group = (group[1],)
            else:
                group = group[1]
        dtype = None
        if isinstance(node, ir.ComputedBuffer):
            dtype = node.data.dtype
        metadata = TensorMetadata(group, dtype, None, None, None, None, None)
        node.meta['tensor_meta'] = metadata
    if print_graph:
        print(graph)
    gm = GraphModule({}, graph)
    legalize_graph(gm)
    gm.graph.lint()
    draw_graph(gm, fname, clear_meta=False, dot_graph_shape=config.trace.dot_graph_shape)