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
def is_smaller(s1: PythonSignature, s2: PythonSignature) -> bool:
    """Returns True if s1 < s2 in the partial order."""
    args1, args2 = (s1.arguments(skip_outputs=True), s2.arguments(skip_outputs=True))
    if len(args1) != len(args2):
        return False
    equal = all((arg1.type == arg2.type for arg1, arg2 in zip(args1, args2)))
    smaller_or_equal = all((str(arg1.type) == str(arg2.type) or is_arg_smaller(arg1.type, arg2.type) for arg1, arg2 in zip(args1, args2)))
    return smaller_or_equal and (not equal)