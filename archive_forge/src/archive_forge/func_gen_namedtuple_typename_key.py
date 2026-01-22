import itertools
import re
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import yaml
from torchgen.api import cpp
from torchgen.api.python import (
from torchgen.code_template import CodeTemplate
from torchgen.context import with_native_function
from torchgen.gen import cpp_string, parse_native_yaml, parse_tags_yaml
from torchgen.model import (
from torchgen.utils import FileManager, split_name_params
from torchgen.yaml_utils import YamlLoader
from .gen_trace_type import should_trace
@with_native_function
def gen_namedtuple_typename_key(f: NativeFunction) -> str:
    name = cpp.name(f.func)
    fieldnames = namedtuple_fieldnames(f.func.returns)
    return '_'.join([name] + fieldnames)