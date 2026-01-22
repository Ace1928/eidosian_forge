import functools
import importlib
import logging
import os
import tempfile
import torch
from .common import device_from_inputs, fake_tensor_unsupported
from .registry import register_backend
def exec_tvm(*i_args):
    args = [a.contiguous() for a in i_args]
    shape_info, _ = m.get_input_info()
    active_inputs = {name for name, _ in shape_info.items()}
    for idx, arg in enumerate(args, 0):
        if arg.dim() != 0:
            if arg.requires_grad:
                arg = arg.detach()
            inp_name = f'inp_{idx}'
            if inp_name not in active_inputs:
                log.warning("input %s skipped as not found in tvm's runtime library", inp_name)
                continue
            m.set_input(inp_name, to_tvm_tensor(arg))
    m.run()
    return [to_torch_tensor(m.get_output(i)) for i in range(m.get_num_outputs())]