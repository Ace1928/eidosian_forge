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
def parse_native_yaml(path: str, tags_yaml_path: str, ignore_keys: Optional[Set[DispatchKey]]=None, *, skip_native_fns_gen: bool=False, loaded_yaml: Optional[object]=None) -> ParsedYaml:
    global _GLOBAL_PARSE_NATIVE_YAML_CACHE
    if path not in _GLOBAL_PARSE_NATIVE_YAML_CACHE:
        valid_tags = parse_tags_yaml(tags_yaml_path)
        if loaded_yaml is None:
            with open(path) as f:
                es = yaml.load(f, Loader=LineLoader)
        else:
            es = loaded_yaml
        _GLOBAL_PARSE_NATIVE_YAML_CACHE[path] = parse_native_yaml_struct(es, valid_tags, ignore_keys, path=path, skip_native_fns_gen=skip_native_fns_gen)
    return _GLOBAL_PARSE_NATIVE_YAML_CACHE[path]