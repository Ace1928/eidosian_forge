import itertools
from contextlib import nullcontext
from functools import partial
from typing import Any
import torch
from torch.utils import benchmark
import xformers.ops.swiglu_op as xsw
from xformers.benchmarks.utils import benchmark_main_helper
def benchmark_swiglu_bw(shape, dtype, bias: bool):
    if dtype == 'autocast_half':
        inp_dtype, model_dtype = (torch.float, torch.float)
        cm: Any = partial(torch.cuda.amp.autocast, enabled=True, dtype=torch.float16)
    else:
        inp_dtype, model_dtype = (dtype, dtype)
        cm = nullcontext
    x = torch.randn(shape[:2], device=device, dtype=inp_dtype)
    x.requires_grad_()
    module = xsw.SwiGLU(in_features=shape[1], hidden_features=shape[2], bias=bias).to(device).to(model_dtype)
    dtype_str = DTYPE2STR.get(dtype, dtype)
    bstr = 'bias' if bias else 'nobi'
    sub_label = f'{dtype_str} B={shape[0]}, I={shape[1]}, H={shape[2]} {bstr}'
    params = module._ordered_params()
    with cm():
        out = xsw.swiglu(x, *params, op=OP)
    grad = torch.zeros_like(out)
    yield benchmark.Timer(stmt='out.backward(grad, retain_graph=True)', globals={'out': out, 'grad': grad}, label='swiglu_bw', description=OP.NAME, sub_label=sub_label)
    del out
    with cm():
        out = xsw.swiglu(x, *params, op=xsw.SwiGLUEagerOp)
    yield benchmark.Timer(stmt='out.backward(grad, retain_graph=True)', globals={'out': out, 'grad': grad}, label='swiglu_bw', description='eager', sub_label=sub_label)