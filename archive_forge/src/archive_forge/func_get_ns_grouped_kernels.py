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
def get_ns_grouped_kernels(*, grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]], backend_indices: Dict[DispatchKey, BackendIndex], native_function_decl_gen: Callable[[Union[NativeFunctionsGroup, NativeFunction], BackendIndex], List[str]]=dest.compute_native_function_declaration) -> Dict[str, List[str]]:
    ns_grouped_kernels: Dict[str, List[str]] = defaultdict(list)
    for f in grouped_native_functions:
        native_function_namespaces = set()
        dispatch_keys = set()
        for dispatch_key, backend_idx in backend_indices.items():
            backend_metadata = backend_idx.get_kernel(f)
            if backend_metadata:
                namespace = backend_metadata.cpp_namespace
                dispatch_keys.add(dispatch_key)
                native_function_namespaces.add(namespace)
            else:
                namespace = DEFAULT_KERNEL_NAMESPACE
            assert len(native_function_namespaces) <= 1, f'Codegen only supports one namespace per operator, got {native_function_namespaces} from {dispatch_keys}'
            ns_grouped_kernels[namespace].extend(native_function_decl_gen(f, backend_idx))
    return ns_grouped_kernels