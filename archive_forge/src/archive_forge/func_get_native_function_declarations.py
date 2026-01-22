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
def get_native_function_declarations(*, grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]], backend_indices: Dict[DispatchKey, BackendIndex], native_function_decl_gen: Callable[[Union[NativeFunctionsGroup, NativeFunction], BackendIndex], List[str]]=dest.compute_native_function_declaration) -> List[str]:
    """
    Generate kernel declarations, in `NativeFunction(s).h`.
    :param grouped_native_functions: a sequence of `NativeFunction` or `NativeFunctionGroup`.
    :param backend_indices: kernel collections grouped by dispatch key.
    :param native_function_decl_gen: callable to generate kernel declaration for each `NativeFunction`.
    :return: a list of string, from the string with all declarations, grouped by namespaces, split by newline.
    """
    ns_grouped_kernels = get_ns_grouped_kernels(grouped_native_functions=grouped_native_functions, backend_indices=backend_indices, native_function_decl_gen=native_function_decl_gen)
    return get_native_function_declarations_from_ns_grouped_kernels(ns_grouped_kernels=ns_grouped_kernels)