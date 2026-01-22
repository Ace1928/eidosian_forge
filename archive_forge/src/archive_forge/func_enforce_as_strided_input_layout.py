from __future__ import annotations
import itertools
import logging
import weakref
from typing import Any, List, Optional, Tuple
import torch
import torch.utils._pytree as pytree
from torch._dynamo.utils import dynamo_timed, lazy_format_graph_code
from torch._functorch.aot_autograd import MutationType
from torch._functorch.compile_utils import fx_graph_cse
from torch._inductor.constant_folding import constant_fold, replace_node_with_constant
from torch._inductor.fx_passes.freezing_patterns import freezing_passes
from torch._inductor.fx_passes.post_grad import view_to_reshape
from . import config
def enforce_as_strided_input_layout(gm: torch.fx.GraphModule):
    """
    Make sure the as_strided node's input's layout does not change due to compiler
    optimizations, because the as_strided strides info depends on input tensor stride info.
    """
    as_strided_ops = [torch.ops.aten.as_strided.default, torch.ops.aten.as_strided_.default, torch.ops.aten.as_strided_scatter.default]
    strided_nodes = [n for n in gm.graph.nodes if n.target in as_strided_ops]
    for n in strided_nodes:
        with gm.graph.inserting_before(n):
            ft = n.args[0].meta['val']
            new_node = gm.graph.call_function(prims.inductor_force_stride_order.default, (n.args[0], ft.stride()))
            n.replace_input_with(n.args[0], new_node)
    gm.graph.lint()
    gm.recompile()