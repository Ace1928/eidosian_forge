import contextlib
import functools
import logging
from unittest.mock import patch
import torch
from torch._dynamo import disable
from torch._dynamo.utils import counters, defake
from torch._functorch.aot_autograd import aot_module_simplified
from torch.utils._python_dispatch import _disable_current_modes
def compiler_fn(gm: torch.fx.GraphModule, example_inputs):
    if callable(kwargs.get('decompositions')):
        kwargs['decompositions'] = kwargs['decompositions']()
    counters['aot_autograd']['total'] += 1
    use_fallback = False
    if use_fallback:
        log.debug('Unable to use AOT Autograd because graph has mutation')
        counters['aot_autograd']['not_ok'] += 1
        return gm

    def _wrapped_bw_compiler(*args, **kwargs):
        return disable(disable(bw_compiler)(*args, **kwargs))
    bw_compiler = kwargs.get('bw_compiler') or kwargs['fw_compiler']
    kwargs['bw_compiler'] = _wrapped_bw_compiler
    kwargs['inference_compiler'] = kwargs.get('inference_compiler') or kwargs['fw_compiler']
    from functorch.compile import nop
    from torch._inductor.debug import enable_aot_logging
    if kwargs.get('fw_compiler', None) == nop:
        patch_config = patch('functorch.compile.config.debug_assert', True)
    else:
        patch_config = contextlib.nullcontext()
    try:
        with enable_aot_logging(), patch_config:
            cg = aot_module_simplified(gm, example_inputs, **kwargs)
            counters['aot_autograd']['ok'] += 1
            return disable(cg)
    except Exception:
        counters['aot_autograd']['not_ok'] += 1
        raise