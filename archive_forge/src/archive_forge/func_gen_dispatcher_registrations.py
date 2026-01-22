import argparse
import os
import pathlib
import re
from collections import Counter, defaultdict, namedtuple
from typing import Dict, List, Optional, Sequence, Set, Union
import yaml
import torchgen.api.dispatcher as dispatcher
import torchgen.dest as dest
from torchgen.api.types import DispatcherSignature
from torchgen.code_template import CodeTemplate
from torchgen.context import native_function_manager
from torchgen.gen import get_grouped_native_functions, parse_native_yaml
from torchgen.model import (
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import concatMap, context, FileManager, NamespaceHelper, Target
from torchgen.yaml_utils import YamlLoader
def gen_dispatcher_registrations(fm: FileManager, output_dir: str, class_name: str, backend_indices: Dict[DispatchKey, BackendIndex], grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]], backend_dispatch_key: DispatchKey, dispatch_key: DispatchKey, selector: 'SelectiveBuilder', build_in_tree: bool=False, per_operator_headers: bool=False, backend_name: str='', eager_registration: bool=True) -> None:
    headers = [f'{output_dir}/{backend_dispatch_key}NativeFunctions.h']
    if build_in_tree:
        external_backend_headers_str = '\n'.join((f'#include <{h}>' for h in headers))
    else:
        external_backend_headers_str = '\n'.join((f'#include "{h}"' for h in headers))
    assert class_name is not None
    backend_index = backend_indices[dispatch_key]
    dispatch_registrations_body = list(concatMap(dest.RegisterDispatchKey(backend_index, Target.REGISTRATION, selector, rocm=False, symint=True, class_method_name=f'{class_name}', skip_dispatcher_op_registration=False), grouped_native_functions))
    newline = '\n'
    ns_helper = NamespaceHelper(namespace_str='at')
    deferred_dispatch_registrations = ''
    static_init_dispatch_registrations = ''
    if eager_registration:
        static_template = CodeTemplate('TORCH_LIBRARY_IMPL(aten, $dispatch_key, m) {\n    $dispatch_registrations_body\n};')
        static_init_dispatch_registrations = static_template.substitute(dispatch_key=dispatch_key, dispatch_registrations_body=dispatch_registrations_body)
    else:
        deferred_template = CodeTemplate('TORCH_API void Register${backend_name}${dispatch_key}NativeFunctions();\nTORCH_API void Register${backend_name}${dispatch_key}NativeFunctions() {\n    static auto m = MAKE_TORCH_LIBRARY_IMPL(aten, $dispatch_key);\n    $dispatch_registrations_body\n}')
        deferred_dispatch_registrations = deferred_template.substitute(backend_name=backend_name, dispatch_key=dispatch_key, dispatch_registrations_body=dispatch_registrations_body)
    fm.write_with_template(f'Register{dispatch_key}.cpp', 'RegisterDispatchKey.cpp', lambda: {'extra_cuda_headers': '', 'external_backend_headers': external_backend_headers_str, 'ops_headers': '#include <ATen/Functions.h>' if not per_operator_headers else '', 'DispatchKey': dispatch_key, 'dispatch_namespace': dispatch_key.lower(), 'dispatch_headers': dest.gen_registration_headers(backend_index, per_operator_headers=per_operator_headers, rocm=False), 'dispatch_definitions': fm.substitute_with_template('RegisterDispatchDefinitions.ini', lambda: {'ns_prologue': ns_helper.prologue, 'ns_epilogue': ns_helper.epilogue, 'static_init_dispatch_registrations': static_init_dispatch_registrations, 'deferred_dispatch_registrations': deferred_dispatch_registrations, 'dispatch_helpers': dest.gen_registration_helpers(backend_index), 'dispatch_namespace': dispatch_key.lower(), 'dispatch_namespaced_definitions': '', 'dispatch_anonymous_definitions': list(concatMap(dest.RegisterDispatchKey(backend_index, Target.ANONYMOUS_DEFINITION, selector, rocm=False, symint=True, class_method_name=f'{class_name}', skip_dispatcher_op_registration=False), grouped_native_functions))}).split(newline)})