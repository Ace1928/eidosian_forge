from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
from torchgen import dest
from torchgen.api.types import DispatcherSignature  # isort:skip
from torchgen.context import method_with_native_function
from torchgen.executorch.model import ETKernelIndex
from torchgen.model import DispatchKey, NativeFunction, Variant
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import concatMap, Target
def gen_custom_ops_registration(*, native_functions: Sequence[NativeFunction], selector: SelectiveBuilder, kernel_index: ETKernelIndex, rocm: bool) -> Tuple[str, str]:
    """
    Generate custom ops registration code for dest.RegisterDispatchKey.

    :param native_functions: a sequence of `NativeFunction`
    :param selector: for selective build.
    :param kernel_index: kernels for all the ops.
    :param rocm: bool for dest.RegisterDispatchKey.
    :return: generated C++ code to register custom operators into PyTorch
    """
    dispatch_key = DispatchKey.CPU
    backend_index = kernel_index._to_backend_index()
    static_init_dispatch_registrations = ''
    ns_grouped_native_functions: Dict[str, List[NativeFunction]] = defaultdict(list)
    for native_function in native_functions:
        ns_grouped_native_functions[native_function.namespace].append(native_function)
    for namespace, functions in ns_grouped_native_functions.items():
        if len(functions) == 0:
            continue
        dispatch_registrations_body = '\n'.join(list(concatMap(dest.RegisterDispatchKey(backend_index, Target.REGISTRATION, selector, rocm=rocm, symint=False, class_method_name=None, skip_dispatcher_op_registration=False), functions)))
        static_init_dispatch_registrations += f'\nTORCH_LIBRARY_IMPL({namespace}, {dispatch_key}, m) {{\n{dispatch_registrations_body}\n}};'
    anonymous_definition = '\n'.join(list(concatMap(dest.RegisterDispatchKey(backend_index, Target.ANONYMOUS_DEFINITION, selector, rocm=rocm, symint=False, class_method_name=None, skip_dispatcher_op_registration=False), native_functions)))
    return (anonymous_definition, static_init_dispatch_registrations)