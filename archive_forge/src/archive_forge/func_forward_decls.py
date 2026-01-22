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
def forward_decls(name: BaseOperatorName, overloads: Sequence[PythonSignatureNativeFunctionPair], *, method: bool) -> Tuple[str, ...]:
    if method:
        return ()
    pycname = get_pycname(name)
    if is_noarg(overloads):
        return (f'static PyObject * {pycname}(PyObject* self_, PyObject* args);\n',)
    else:
        return (f'static PyObject * {pycname}(PyObject* self_, PyObject* args, PyObject* kwargs);\n',)