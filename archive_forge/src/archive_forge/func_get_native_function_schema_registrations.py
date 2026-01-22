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
def get_native_function_schema_registrations(*, native_functions: Sequence[NativeFunction], schema_selector: SelectiveBuilder) -> Tuple[List[str], str]:
    ns_native_functions: Dict[str, List[NativeFunction]] = defaultdict(list)
    for native_function in native_functions:
        ns_native_functions[native_function.namespace].append(native_function)
    schema_registrations = ''
    aten_schema_registrations = []
    custom_namespace = None
    for namespace, funcs in ns_native_functions.items():
        schema_registrations_body = list(mapMaybe(RegisterSchema(schema_selector), funcs))
        if namespace == 'aten':
            aten_schema_registrations = schema_registrations_body
        else:
            custom_namespace = namespace
            tab = '\t'
            torch_library_macro = 'TORCH_LIBRARY_FRAGMENT' if namespace in FRAGMENT_NAMESPACES else 'TORCH_LIBRARY'
            schema_registrations += f'\n{torch_library_macro}({custom_namespace}, m) {{\n  {tab.join(schema_registrations_body)}\n}};'
    return (aten_schema_registrations, schema_registrations)