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
def get_grouped_by_view_native_functions(native_functions: Sequence[NativeFunction]) -> Sequence[Union[NativeFunction, NativeFunctionsViewGroup]]:

    def maybe_create_view_group(d: Dict[Union[ViewSchemaKind, SchemaKind], NativeFunction]) -> List[Union[NativeFunction, NativeFunctionsViewGroup]]:
        funcs: List[Union[NativeFunction, NativeFunctionsViewGroup]] = []
        if ViewSchemaKind.aliasing in d:
            view = d.pop(ViewSchemaKind.aliasing)
            view_inplace = d.pop(ViewSchemaKind.aliasing_inplace, None)
            view_copy = d.pop(SchemaKind.functional, None)
            funcs.append(NativeFunctionsViewGroup(view=view, view_copy=view_copy, view_inplace=view_inplace))
        for func in d.values():
            funcs.append(func)
        return funcs
    grouped_by_views: Dict[FunctionSchema, Dict[Union[SchemaKind, ViewSchemaKind], NativeFunction]] = defaultdict(dict)
    for f in native_functions:
        schema = f.func.view_signature()
        view_kind: ViewSchemaKind = f.view_schema_kind
        if view_kind == ViewSchemaKind.non_aliasing:
            kind = f.func.kind()
            assert kind not in grouped_by_views[schema]
            grouped_by_views[schema][kind] = f
        else:
            assert view_kind not in grouped_by_views[schema]
            grouped_by_views[schema][view_kind] = f
    return list(concatMap(maybe_create_view_group, grouped_by_views.values()))