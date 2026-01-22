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
def find_required_inputs(formula: str, postfix: str) -> Tuple[str, ...]:
    is_foreach = f.func.name.name.base.startswith('_foreach_')
    required_inputs = set()
    for arg in args_with_derivatives:
        if arg.type in ('at::TensorList', 'const at::ITensorListRef &') and (not is_foreach):
            continue
        arg_name = arg.name
        found = re.search(IDENT_REGEX.format(arg_name), formula)
        if found:
            raise RuntimeError(f'The forward formula for {defn_name} is using the base name of the {arg_name} argument which is ambiguous. You should use {arg_name}_p to access the primal value and {arg_name}_t to access the tangent.')
        found = re.search(IDENT_REGEX.format(arg_name + postfix), formula)
        if found:
            required_inputs.add(arg_name)
    return tuple(required_inputs)