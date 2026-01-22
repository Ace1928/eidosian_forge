import re
from collections import defaultdict
from typing import Any, Counter, Dict, List, Match, Optional, Sequence, Set, Tuple
import yaml
from torchgen.api import cpp
from torchgen.api.autograd import (
from torchgen.api.types import (
from torchgen.context import with_native_function
from torchgen.gen import get_grouped_by_view_native_functions, parse_native_yaml
from torchgen.model import (
from torchgen.utils import concatMap, IDENT_REGEX, split_name_params
from torchgen.yaml_utils import YamlLoader
def canonical_function(functions: Sequence[NativeFunction], name: str) -> NativeFunction:
    for f in functions:
        if not f.func.is_functional_fn() and (not f.func.is_out_fn()) and (name == str(f.func.name.name)):
            return f
    assert name + '_' == cpp.name(functions[0].func)
    return functions[0]