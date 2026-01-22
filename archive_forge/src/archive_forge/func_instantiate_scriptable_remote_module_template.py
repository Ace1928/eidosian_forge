import importlib
import logging
import os
import sys
import tempfile
from typing import Optional
import torch
from torch.distributed.nn.jit.templates.remote_module_template import (
def instantiate_scriptable_remote_module_template(module_interface_cls, enable_moving_cpu_tensors_to_cuda=True):
    if not getattr(module_interface_cls, '__torch_script_interface__', False):
        raise ValueError(f'module_interface_cls {module_interface_cls} must be a type object decorated by @torch.jit.interface')
    module_interface_cls_name = torch._jit_internal._qualified_name(module_interface_cls).replace('.', '_')
    generated_module_name = f'{_FILE_PREFIX}{module_interface_cls_name}'
    assign_module_interface_cls_str = f'from {module_interface_cls.__module__} import {module_interface_cls.__name__} as module_interface_cls'
    args_str, arg_types_str, return_type_str = get_arg_return_types_from_interface(module_interface_cls)
    kwargs_str = ''
    arrow_and_return_type_str = f' -> {return_type_str}'
    arrow_and_future_return_type_str = f' -> Future[{return_type_str}]'
    str_dict = dict(assign_module_interface_cls=assign_module_interface_cls_str, arg_types=arg_types_str, arrow_and_return_type=arrow_and_return_type_str, arrow_and_future_return_type=arrow_and_future_return_type_str, args=args_str, kwargs=kwargs_str, jit_script_decorator='@torch.jit.script')
    return _do_instantiate_remote_module_template(generated_module_name, str_dict, enable_moving_cpu_tensors_to_cuda)