import argparse
import functools
import json
import os
import pathlib
from collections import defaultdict, namedtuple, OrderedDict
from dataclasses import dataclass, field
from typing import (
import yaml
import torchgen.api.dispatcher as dispatcher
import torchgen.api.meta as meta
import torchgen.api.native as native
import torchgen.api.structured as structured
import torchgen.dest as dest
from torchgen.api import cpp
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.context import (
from torchgen.gen_functionalization_type import (
from torchgen.gen_vmap_plumbing import gen_all_vmap_plumbing
from torchgen.model import (
from torchgen.native_function_generation import (
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import (
from torchgen.yaml_utils import YamlDumper, YamlLoader
def gen_per_operator_headers(*, native_functions: Sequence[NativeFunction], grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]], static_dispatch_idx: List[BackendIndex], selector: SelectiveBuilder, backend_indices: Dict[DispatchKey, BackendIndex], cpu_fm: FileManager, cuda_fm: FileManager, ops_fm: FileManager, functions_keys: Set[DispatchKey], dispatch_keys: Sequence[DispatchKey], rocm: bool) -> None:
    functions_by_root_name: Dict[str, List[NativeFunction]] = defaultdict(list)
    for fn in native_functions:
        functions_by_root_name[fn.root_name].append(fn)
    grouped_functions_by_root_name: Dict[str, List[Union[NativeFunction, NativeFunctionsGroup]]] = defaultdict(list)
    for group in grouped_native_functions:
        name = group.root_name
        grouped_functions_by_root_name[name].append(group)
    for name, functions in functions_by_root_name.items():
        ops_fm.write_with_template(f'{name}_ops.h', 'Operator.h', lambda: {'declarations': list(mapMaybe(ComputeOperators(Target.DECLARATION, static_dispatch_backend_indices=static_dispatch_idx), functions))})
        ops_fm.write_with_template(f'{name}.h', 'Function.h', lambda: {'static_dispatch_ops_headers': list(mapMaybe(lambda fn: static_dispatch_ops_header(fn, backend_index=static_dispatch_idx), functions)), 'operator_includes': f'#include <ATen/ops/{name}_ops.h>', 'function_definitions': list(mapMaybe(ComputeFunction(), functions))})
        grouped_functions = grouped_functions_by_root_name.get(name, [])
        structured_functions = [fn for fn in grouped_functions if isinstance(fn, NativeFunctionsGroup) and fn.structured]
        is_structured = len(structured_functions) > 0
        if is_structured:
            ops_fm.write_with_template(f'{name}_meta.h', 'NativeMetaFunction.h', lambda: {'meta_function_declarations': list(mapMaybe(compute_meta_function_declaration, structured_functions))})
        declarations = get_native_function_declarations(grouped_native_functions=grouped_functions, backend_indices=backend_indices, native_function_decl_gen=dest.compute_native_function_declaration)
        ops_fm.write_with_template(f'{name}_native.h', 'NativeFunction.h', lambda: {'extra_includes': f'#include <ATen/ops/{name}_meta.h>' if is_structured else [], 'native_function_declarations': declarations})
    for category, suffix in [('Functions', ''), ('Operators', '_ops'), ('NativeMetaFunctions', '_meta'), ('NativeFunctions', '_native')]:
        cpu_fm.write(f'{category}.h', lambda: {f'{category}_includes': [f'#include <ATen/ops/{name}{suffix}.h>' for name in sorted(functions_by_root_name.keys())], f'{category}_declarations': []})
    for dispatch_key in dispatch_keys:
        if dispatch_key not in functions_keys:
            continue
        dispatch_namespace = dispatch_key.lower()
        dispatch_names = []
        for name, functions in functions_by_root_name.items():
            grouped_functions = grouped_functions_by_root_name.get(name, [])
            declarations = list(concatMap(dest.RegisterDispatchKey(backend_indices[dispatch_key], Target.NAMESPACED_DECLARATION, selector, rocm=rocm, symint=True, class_method_name=None, skip_dispatcher_op_registration=False), grouped_functions))
            if len(declarations) == 0:
                continue
            dispatch_names.append(name)
            ops_fm.write_with_template(f'{name}_{dispatch_namespace}_dispatch.h', 'DispatchKeyFunction.h', lambda: {'dispatch_namespace': dispatch_namespace, 'dispatch_namespaced_declarations': declarations})
        fm = cuda_fm if is_cuda_dispatch_key(dispatch_key) else cpu_fm
        inl_headers = f'#include <ATen/{dispatch_key}Functions_inl.h>'
        fm.write_with_template(f'{dispatch_key}Functions.h', 'DispatchKeyFunctions.h', lambda: {'dispatch_key': str(dispatch_key), 'inline_headers': inl_headers})
        fm.write_with_template(f'{dispatch_key}Functions_inl.h', 'DispatchKeyFunctions_inl.h', lambda: {'dispatch_namespace': dispatch_namespace, 'DispatchKeyFunctions_inl_includes': [f'#include <ATen/ops/{name}_{dispatch_namespace}_dispatch.h>' for name in sorted(dispatch_names)], 'dispatch_namespaced_declarations': []})
        del fm
    cpu_fm.write('MethodOperators.h', lambda: {'MethodOperators_includes': sorted((f'#include <ATen/ops/{name}_ops.h>' for name, functions in functions_by_root_name.items() if any((Variant.method in fn.variants for fn in functions)))), 'MethodOperators_declarations': []})