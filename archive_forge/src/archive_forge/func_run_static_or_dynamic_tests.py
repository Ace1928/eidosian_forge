import torch
from .make_fx import make_fx_check
from .aot_autograd import aot_autograd_check
from .fake_tensor import fake_check
from torch._subclasses.schema_check_mode import SchemaCheckMode
def run_static_or_dynamic_tests(dynamic):
    tracing_mode = 'symbolic' if dynamic else 'fake'
    make_fx_check(func, args, kwargs, tracing_mode=tracing_mode)
    if supports_autograd:
        aot_autograd_check(func, args, kwargs, dynamic=dynamic)
    check_compile(func, args, kwargs, fullgraph=fullgraph, backend='aot_eager', dynamic=dynamic)
    check_compile(func, args, kwargs, fullgraph=fullgraph, backend='inductor', dynamic=dynamic)