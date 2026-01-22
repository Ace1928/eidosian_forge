from operator import itemgetter
from functorch.compile import make_boxed_func
import torch
import torch.nn as nn
from torch._functorch.compilers import aot_module
from torch._inductor.decomposition import select_decomp_table
from torch.distributed._tensor import DTensor
def get_inductor_decomp_graphs(model: nn.Module, args, kwargs):
    """
    Obtain forward and backward graphs of a model with inductor decompositions using tracing and aot_module.

    Convenient util to get the fwd and bwd graphs of an arbitrary model
    with inductor decompositions. Note that this would simply do tracing
    with aot_module and don't ensure correctness. This is useful to track
    the ops needed in DTensor.
    """
    compiled_mod = aot_module(model, fw_compiler=fwd_bwd_compiler, decompositions=inductor_decomps)
    output = compiled_mod(*args, **kwargs)
    if output.ndim != 0:
        output = output.sum()
    output.backward()
    assert len(graphs) == 2
    return graphs