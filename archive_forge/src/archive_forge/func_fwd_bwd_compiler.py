from operator import itemgetter
from functorch.compile import make_boxed_func
import torch
import torch.nn as nn
from torch._functorch.compilers import aot_module
from torch._inductor.decomposition import select_decomp_table
from torch.distributed._tensor import DTensor
def fwd_bwd_compiler(fx_g, _):
    graphs.append(fx_g)
    return make_boxed_func(fx_g)