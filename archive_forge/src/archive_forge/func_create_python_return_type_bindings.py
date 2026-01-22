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
def create_python_return_type_bindings(fm: FileManager, pairs: Sequence[PythonSignatureNativeFunctionPair], pred: Callable[[NativeFunction], bool], filename: str) -> None:
    """
    Generate function to initialize and return named tuple for native functions
    which returns named tuple and registration invocations in `python_return_types.cpp`.
    """
    py_return_types_definition: List[str] = []
    py_return_types_registrations: List[str] = []
    grouped = group_filter_overloads(pairs, pred)
    for name in sorted(grouped.keys(), key=str):
        overloads = grouped[name]
        definitions, registrations = generate_return_type_definition_and_registrations(overloads)
        py_return_types_definition.append('' if not definitions else '\n'.join(definitions))
        py_return_types_registrations.append('' if not registrations else '\n'.join(registrations))
    fm.write_with_template(filename, filename, lambda: {'generated_comment': '@' + f'generated from {fm.template_dir_for_comments()}/{filename}', 'py_return_types': py_return_types_definition, 'py_return_types_registrations': py_return_types_registrations})