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
def get_native_function_declarations_from_ns_grouped_kernels(*, ns_grouped_kernels: Dict[str, List[str]]) -> List[str]:
    declarations: List[str] = []
    newline = '\n'
    for namespace, kernels in ns_grouped_kernels.items():
        ns_helper = NamespaceHelper(namespace_str=namespace, entity_name='', max_level=4)
        ordered_kernels = list(OrderedDict.fromkeys(kernels))
        declarations.extend(f'\n{ns_helper.prologue}\n{newline.join(ordered_kernels)}\n{ns_helper.epilogue}\n        '.split(newline))
    return declarations