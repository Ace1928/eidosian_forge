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
def create_python_bindings_sharded(fm: FileManager, pairs: Sequence[PythonSignatureNativeFunctionPair], pred: Callable[[NativeFunction], bool], module: Optional[str], filename: str, *, method: bool, num_shards: int, symint: bool=True) -> None:
    """Generates Python bindings to ATen functions"""
    grouped = group_filter_overloads(pairs, pred)

    def key_func(kv: Tuple[BaseOperatorName, List[PythonSignatureNativeFunctionPair]]) -> str:
        return kv[0].base

    def env_func(kv: Tuple[BaseOperatorName, List[PythonSignatureNativeFunctionPair]]) -> Dict[str, List[str]]:
        name, fn_pairs = kv
        return {'ops_headers': [f'#include <ATen/ops/{name.base}.h>'], 'py_forwards': list(forward_decls(name, fn_pairs, method=method)), 'py_methods': [method_impl(name, module, fn_pairs, method=method, symint=symint)], 'py_method_defs': [method_def(name, module, fn_pairs, method=method)]}
    fm.write_sharded(filename, grouped.items(), base_env={'generated_comment': '@' + f'generated from {fm.template_dir_for_comments()}/{filename}'}, key_fn=key_func, env_callable=env_func, num_shards=num_shards, sharded_keys={'ops_headers', 'py_forwards', 'py_methods', 'py_method_defs'})