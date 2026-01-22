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
def parse_native_yaml_struct(es: object, valid_tags: Set[str], ignore_keys: Optional[Set[DispatchKey]]=None, path: str='<stdin>', skip_native_fns_gen: bool=False) -> ParsedYaml:
    assert isinstance(es, list)
    rs: List[NativeFunction] = []
    bs: Dict[DispatchKey, Dict[OperatorName, BackendMetadata]] = defaultdict(dict)
    for e in es:
        assert isinstance(e.get('__line__'), int), e
        loc = Location(path, e['__line__'])
        funcs = e.get('func')
        with context(lambda: f'in {loc}:\n  {funcs}'):
            func, m = NativeFunction.from_yaml(e, loc, valid_tags, ignore_keys)
            rs.append(func)
            BackendIndex.grow_index(bs, m)
    error_check_native_functions(rs)
    indices: Dict[DispatchKey, BackendIndex] = defaultdict(lambda: BackendIndex(dispatch_key=DispatchKey.Undefined, use_out_as_primary=True, external=False, device_guard=False, index={}))
    if not skip_native_fns_gen:
        add_generated_native_functions(rs, bs)
    for k, v in bs.items():
        indices[k] = BackendIndex(dispatch_key=k, use_out_as_primary=True, external=False, device_guard=is_cuda_dispatch_key(k), index=v)
    return ParsedYaml(rs, indices)