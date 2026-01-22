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
def load_signatures(native_functions: List[NativeFunction], deprecated_yaml_path: str, *, method: bool, skip_deprecated: bool=False, pyi: bool=False) -> Sequence[PythonSignatureNativeFunctionPair]:

    @with_native_function
    def gen_signature_pairs(f: NativeFunction) -> PythonSignatureNativeFunctionPair:
        return PythonSignatureNativeFunctionPair(signature=signature(f, method=method, pyi=pyi), function=f)
    pairs = list(map(gen_signature_pairs, native_functions))
    deprecated = load_deprecated_signatures(pairs, deprecated_yaml_path, method=method, pyi=pyi)
    return pairs if skip_deprecated else pairs + deprecated