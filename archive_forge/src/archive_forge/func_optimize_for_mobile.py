import torch
from enum import Enum
from torch._C import _MobileOptimizerType as MobileOptimizerType
from typing import Optional, Set, List, AnyStr
def optimize_for_mobile(script_module: torch.jit.ScriptModule, optimization_blocklist: Optional[Set[MobileOptimizerType]]=None, preserved_methods: Optional[List[AnyStr]]=None, backend: str='CPU') -> torch.jit.RecursiveScriptModule:
    """
    Optimize a torch script module for mobile deployment.

    Args:
        script_module: An instance of torch script module with type of ScriptModule.
        optimization_blocklist: A set with type of MobileOptimizerType. When set is not passed,
            optimization method will run all the optimizer pass; otherwise, optimizer
            method will run the optimization pass that is not included inside optimization_blocklist.
        preserved_methods: A list of methods that needed to be preserved when freeze_module pass is invoked
        backend: Device type to use for running the result model ('CPU'(default), 'Vulkan' or 'Metal').
    Returns:
        A new optimized torch script module
    """
    if not isinstance(script_module, torch.jit.ScriptModule):
        raise TypeError(f'Got {type(script_module)}, but ScriptModule is expected.')
    if optimization_blocklist is None:
        optimization_blocklist = set()
    if preserved_methods is None:
        preserved_methods = []
    preserved_methods_str: List[str] = [str(method) for method in preserved_methods]
    bundled_inputs_attributes = _get_bundled_inputs_preserved_attributes(script_module, preserved_methods_str)
    if all((hasattr(script_module, method) for method in bundled_inputs_attributes)):
        preserved_methods_str = list(set(preserved_methods_str + bundled_inputs_attributes))
    non_exist_methods = []
    for method in preserved_methods_str:
        if not hasattr(script_module, method):
            non_exist_methods.append(method)
    if non_exist_methods:
        raise AttributeError(f'The following methods to preserve do not exist in script_module: {', '.join(non_exist_methods)}')
    backend = backend.lower()
    if backend == 'cpu':
        optimized_cpp_module = torch._C._jit_pass_optimize_for_mobile(script_module._c, optimization_blocklist, preserved_methods_str)
    elif backend == 'vulkan':
        optimized_cpp_module = torch._C._jit_pass_vulkan_optimize_for_mobile(script_module._c, optimization_blocklist, preserved_methods_str)
    elif backend == 'metal':
        optimized_cpp_module = torch._C._jit_pass_metal_optimize_for_mobile(script_module._c, preserved_methods_str)
    else:
        raise TypeError("Unknown backend, must be one of 'CPU', 'Vulkan' or 'Metal'")
    return torch.jit._recursive.wrap_cpp_module(optimized_cpp_module)