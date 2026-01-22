import contextlib
import copy
import functools
import inspect
import os
import re
import warnings
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar
from typing_extensions import ParamSpec
import torch
from torch._jit_internal import (
from torch.autograd import function
from torch.jit._script import _CachedForward, script, ScriptModule
from torch.jit._state import _enabled, _python_cu
from torch.nn import Module
from torch.testing._comparison import default_tolerances
def trace_module(mod, inputs, optimize=None, check_trace=True, check_inputs=None, check_tolerance=1e-05, strict=True, _force_outplace=False, _module_class=None, _compilation_unit=_python_cu, example_inputs_is_kwarg=False, _store_inputs=True):
    """
    Trace a module and return an executable :class:`ScriptModule` that will be optimized using just-in-time compilation.

    When a module is passed to :func:`torch.jit.trace <torch.jit.trace>`, only
    the ``forward`` method is run and traced. With ``trace_module``, you can specify a dictionary of
    method names to example inputs to trace (see the ``inputs``) argument below.

    See :func:`torch.jit.trace <torch.jit.trace>` for more information on tracing.

    Args:
        mod (torch.nn.Module):  A ``torch.nn.Module`` containing methods whose names are
                                specified in ``inputs``. The given methods will be compiled
                                as a part of a single `ScriptModule`.
        inputs (dict):  A dict containing sample inputs indexed by method names in ``mod``.
                                The inputs will be passed to methods whose names correspond to inputs'
                                keys while tracing.
                                ``{ 'forward' : example_forward_input, 'method2': example_method2_input}``
    Keyword arguments:
        check_trace (``bool``, optional): Check if the same inputs run through
                                      traced code produce the same outputs. Default: ``True``. You might want
                                      to disable this if, for example, your network contains non-
                                      deterministic ops or if you are sure that the network is correct despite
                                      a checker failure.

        check_inputs (list of dicts, optional): A list of dicts of input arguments that should be used
                                                 to check the trace against what is expected. Each tuple
                                                 is equivalent to a set of input arguments that would
                                                 be specified in ``inputs``. For best results, pass in a
                                                 set of checking inputs representative of the space of
                                                 shapes and types of inputs you expect the network to see.
                                                 If not specified, the original ``inputs`` are used for checking
        check_tolerance (float, optional): Floating-point comparison tolerance to use in the checker procedure.
                                           This can be used to relax the checker strictness in the event that
                                           results diverge numerically for a known reason, such as operator fusion.
        example_inputs_is_kwarg (``bool``, optional): This parameter indicate whether the example inputs is a pack
                                           pack of keyword arguments. Default: ``False``.

    Returns:
        A :class:`ScriptModule` object with a single ``forward`` method containing the traced code.
        When ``func`` is a ``torch.nn.Module``, the returned :class:`ScriptModule` will have the same set of
        sub-modules and parameters as ``func``.

    Example (tracing a module with multiple methods)::

        import torch
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 3)

            def forward(self, x):
                return self.conv(x)

            def weighted_kernel_sum(self, weight):
                return weight * self.conv.weight


        n = Net()
        example_weight = torch.rand(1, 1, 3, 3)
        example_forward_input = torch.rand(1, 1, 3, 3)

        # Trace a specific method and construct `ScriptModule` with
        # a single `forward` method
        module = torch.jit.trace(n.forward, example_forward_input)

        # Trace a module (implicitly traces `forward`) and construct a
        # `ScriptModule` with a single `forward` method
        module = torch.jit.trace(n, example_forward_input)

        # Trace specific methods on a module (specified in `inputs`), constructs
        # a `ScriptModule` with `forward` and `weighted_kernel_sum` methods
        inputs = {'forward' : example_forward_input, 'weighted_kernel_sum' : example_weight}
        module = torch.jit.trace_module(n, inputs)

    """
    if not _enabled:
        return mod
    if optimize is not None:
        warnings.warn('`optimize` is deprecated and has no effect. Use `with torch.jit.optimized_execution() instead')
    var_lookup_fn = _create_interpreter_name_lookup_fn(0)
    if not isinstance(mod, torch.nn.Module):
        raise AttributeError('expected torch.nn.Module as the first argument')
    if not isinstance(inputs, dict):
        raise AttributeError('expected a dictionary of (method_name, input) pairs')
    old_module_map = torch.jit._trace._trace_module_map
    try:
        trace_module_map: Dict[Any, Any] = {}

        def register_submods(mod, prefix):
            for name, child in mod.named_children():
                submod_qualname = prefix + '.' + name
                trace_module_map[child] = submod_qualname
                register_submods(child, submod_qualname)
        trace_module_map['__module'] = mod
        torch.jit._trace._trace_module_map = trace_module_map
        register_submods(mod, '__module')
        module = make_module(mod, _module_class, _compilation_unit)
        for method_name, example_inputs in inputs.items():
            if method_name == 'forward':
                func = mod
                forward_method = getattr(mod, method_name)
                argument_names = get_callable_argument_names(forward_method)
            else:
                func = getattr(mod, method_name)
                argument_names = get_callable_argument_names(func)
            if isinstance(example_inputs, dict) and example_inputs_is_kwarg:
                for key in example_inputs:
                    if key not in argument_names:
                        valid_arguments = '[' + ','.join(argument_names) + ']'
                        raise NameError(f"'{key}' is not in forward() method's arguments,\n                         valid arguments name are {valid_arguments}")
                module._c._create_method_from_trace_with_dict(method_name, func, example_inputs, var_lookup_fn, strict, _force_outplace, argument_names, _store_inputs)
            else:
                example_inputs = make_tuple(example_inputs)
                module._c._create_method_from_trace(method_name, func, example_inputs, var_lookup_fn, strict, _force_outplace, argument_names, _store_inputs)
            check_trace_method = module._c._get_method(method_name)
            if check_trace:
                if check_inputs is not None:
                    _check_trace(check_inputs, func, check_trace_method, check_tolerance, strict, _force_outplace, True, _module_class, example_inputs_is_kwarg=example_inputs_is_kwarg)
                else:
                    _check_trace([inputs], func, check_trace_method, check_tolerance, strict, _force_outplace, True, _module_class, example_inputs_is_kwarg=example_inputs_is_kwarg)
    finally:
        torch.jit._trace._trace_module_map = old_module_map
    return module