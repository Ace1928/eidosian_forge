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
def store_output(self, indices, val, mask):
    """
        Hook called from template code to store the final output
        (if the buffer hasn't been optimized away), then append any
        epilogue fusions.
        """
    assert isinstance(indices, (list, tuple))
    assert isinstance(val, str)
    assert isinstance(mask, str)
    assert self.template_mask is None
    indices = list(map(TritonPrinter.paren, indices))
    index_symbols = [sympy.Symbol(x) for x in indices]
    lengths = [V.graph.sizevars.simplify(s) for s in self.output_node.get_size()]
    assert len(indices) == len(lengths)
    for name, range_tree_entry in zip(indices, self.range_trees[0].construct_entries(lengths)):
        range_tree_entry.set_name(name)
    contiguous_index = sympy_dot(ir.FlexibleLayout.contiguous_strides(lengths), index_symbols)
    contiguous_index = self.rename_indexing(contiguous_index)
    self.body.writeline('xindex = ' + texpr(contiguous_index))
    self.range_trees[0].lookup(sympy.Integer(1), sympy_product(lengths)).set_name('xindex')
    self.template_mask = mask
    self.template_indices = indices
    output_index = self.output_node.get_layout().make_indexer()(index_symbols)
    output_index = self.rename_indexing(output_index)
    if output_index == contiguous_index:
        output_index = sympy.Symbol('xindex')
    epilogue_args = [val]
    for input_node in itertools.chain(self.input_nodes[:self.prefix_args], self.input_nodes[len(self.input_nodes) - self.suffix_args:]):
        input_node.freeze_layout()
        epilogue_args.append(input_node.make_loader()(index_symbols))
    V.ops.store(self.output_node.get_name(), output_index, self.epilogue_fn(*epilogue_args))
    self.codegen_body()

    def hook():
        self.codegen_body()
        return textwrap.indent(self.body.getvalue(), '    ').strip()
    assert '<STORE_OUTPUT>' not in self.render_hooks
    self.render_hooks['<STORE_OUTPUT>'] = hook
    return '<STORE_OUTPUT>'