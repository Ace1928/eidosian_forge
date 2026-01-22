import collections
import functools
import inspect
import sys
import textwrap
import types
import warnings
from typing import Dict, List, Set, Type
import torch
import torch._jit_internal as _jit_internal
from torch._sources import fake_range
from torch.jit._builtins import _find_builtin
from torch.jit._check import AttributeTypeIsSupportedChecker
from torch.jit._state import _add_script_class, _get_script_class, _python_cu
from torch.jit.frontend import (
from torch.nn import Module
def infer_concrete_type_builder(nn_module, share_types=True):
    """
    Build a ConcreteModuleTypeBuilder from an nn.Module.

    This ConcreteModuleType doesn't have a JIT type associated with it yet, it
    must be filled in by the caller.
    """
    concrete_type_builder = torch._C.ConcreteModuleTypeBuilder(type(nn_module))
    if isinstance(nn_module, torch.nn.ModuleDict):
        concrete_type_builder.set_module_dict()
    if isinstance(nn_module, (torch.nn.ModuleList, torch.nn.Sequential)):
        concrete_type_builder.set_module_list()
    if isinstance(nn_module, torch.nn.ParameterList):
        concrete_type_builder.set_parameter_list()
    if isinstance(nn_module, torch.nn.ParameterDict):
        concrete_type_builder.set_parameter_dict()
    class_annotations = get_annotations(nn_module)
    if isinstance(nn_module, torch.ao.quantization.QuantWrapper):
        class_annotations = {}
    user_annotated_ignored_attributes = getattr(nn_module, '__jit_ignored_attributes__', list())
    concrete_type_builder.add_ignored_attributes(user_annotated_ignored_attributes)
    ignored_properties = jit_ignored_properties(nn_module)

    def infer_type(name, item):
        inferred = False
        try:
            if name in class_annotations and class_annotations[name] != torch.nn.Module.__annotations__['forward']:
                ann_to_type = torch.jit.annotations.ann_to_type(class_annotations[name], fake_range())
                attr_type = torch._C.InferredType(ann_to_type)
            elif isinstance(item, torch.jit.Attribute):
                ann_to_type = torch.jit.annotations.ann_to_type(item.type, fake_range())
                attr_type = torch._C.InferredType(ann_to_type)
            else:
                attr_type = torch._C._jit_try_infer_type(item)
                inferred = True
        except RuntimeError as re:
            raise RuntimeError(f'Error inferring type for {name}: {item}: {re}') from re
        return (attr_type, inferred)
    added_names = set()
    for name, item in nn_module._parameters.items():
        if name in user_annotated_ignored_attributes:
            continue
        assert item is None or isinstance(item, torch.Tensor)
        attr_type, _ = infer_type(name, item)
        concrete_type_builder.add_attribute(name, attr_type.type(), True, False)
        added_names.add(name)
    for name, item in nn_module._buffers.items():
        if name in user_annotated_ignored_attributes:
            continue
        assert item is None or isinstance(item, torch.Tensor)
        attr_type, _ = infer_type(name, item)
        concrete_type_builder.add_attribute(name, attr_type.type(), False, True)
        added_names.add(name)
    for name, item in nn_module._modules.items():
        if name in user_annotated_ignored_attributes:
            continue
        attr_type, _ = infer_type(name, item)
        if item is None:
            concrete_type_builder.add_attribute(name, attr_type.type(), False, False)
            continue
        if attr_type.success():
            assert attr_type.type().is_interface_type()
            sub_concrete_type = torch._C.ConcreteModuleType.from_jit_type(attr_type.type())
        else:
            sub_concrete_type = get_module_concrete_type(item, share_types)
        concrete_type_builder.add_module(name, sub_concrete_type)
        added_names.add(name)
    constants_set = set(getattr(nn_module, '__constants__', ()))
    for name, ann in class_annotations.items():
        if torch._jit_internal.is_final(ann):
            constants_set.add(name)
    for name in constants_set:
        if name in added_names:
            if name in nn_module._modules:
                hint = 'submodule'
            elif name in nn_module._buffers:
                hint = 'buffer'
            elif name in nn_module._parameters:
                hint = 'parameter'
            else:
                raise AssertionError('added_names must be submodule, parameter, or buffer')
            warnings.warn(f"'{name}' was found in ScriptModule constants,  but it is a non-constant {hint}. Consider removing it.")
            continue
        if not hasattr(nn_module, name):
            warnings.warn(f"'{name}' was found in ScriptModule constants, but was not actually set in __init__. Consider removing it.")
            continue
        value = getattr(nn_module, name)
        concrete_type_builder.add_constant(name, _get_valid_constant(name, value, type(nn_module).__name__))
        added_names.add(name)
    overloads = getattr(nn_module, '__overloads__', {})
    overloads.update(get_overload_name_mapping(get_overload_annotations(nn_module, ignored_properties)))
    for name, overloaded_names in overloads.items():
        concrete_type_builder.add_overload(name, overloaded_names)
    for name, value in nn_module.__dict__.items():
        if name in ignored_attributes or name.startswith('__'):
            continue
        if name in user_annotated_ignored_attributes:
            continue
        if name in added_names:
            continue
        isoverloadpacket = isinstance(value, torch._ops.OpOverloadPacket)
        if isoverloadpacket:
            value = value.op
        if inspect.isfunction(value):
            try:
                scripted_fn = torch.jit.script(value)
                concrete_type_builder.add_function_attribute(name, torch._C._jit_try_infer_type(scripted_fn).type(), value)
            except Exception as e:
                hint = f'(This function exists as an attribute on the Python module, but we failed to compile it to a TorchScript function. \nThe error stack is reproduced here:\n{e}'
                concrete_type_builder.add_failed_attribute(name, hint)
                pass
            continue
        builtin_symbol_name = _find_builtin(value)
        if builtin_symbol_name:
            concrete_type_builder.add_builtin_function(name, builtin_symbol_name)
            continue
        if isinstance(value, torch.jit.ScriptFunction):
            concrete_type_builder.add_function_attribute(name, torch._C._jit_try_infer_type(value).type(), value)
            continue
        attr_type, inferred = infer_type(name, value)
        if attr_type.success():
            concrete_type_builder.add_attribute(name, attr_type.type(), False, False)
        else:
            inferred_msg = 'Its type was inferred; try adding a type annotation for the attribute.' if inferred else ''
            additional_info = f'{attr_type.reason()}. {inferred_msg}'
            hint = f"(This attribute exists on the Python module, but we failed to convert Python type: '{torch.typename(type(value))}' to a TorchScript type. {additional_info})"
            concrete_type_builder.add_failed_attribute(name, hint)
    for hook in nn_module._forward_hooks.values():
        concrete_type_builder.add_forward_hook(hook)
    for pre_hook in nn_module._forward_pre_hooks.values():
        concrete_type_builder.add_forward_pre_hook(pre_hook)
    return concrete_type_builder