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
def generate_return_type_declarations(overloads: Sequence[PythonSignatureNativeFunctionPair]) -> List[str]:
    """
    Generate block of function declarations in `python_return_types.h` to initialize
    and return named tuple for a native function.
    """
    typenames: Dict[str, str] = {}
    declarations: List[str] = []
    for overload in overloads:
        fieldnames = namedtuple_fieldnames(overload.function.func.returns)
        if not fieldnames:
            continue
        name = cpp.name(overload.function.func)
        tn_key = gen_namedtuple_typename_key(overload.function)
        typename = typenames.get(tn_key)
        if typename is None:
            typename = f'{name}NamedTuple{('' if not declarations else len(declarations))}'
            typenames[tn_key] = typename
            declarations.append(f'PyTypeObject* get_{name}_namedtuple();')
    return declarations