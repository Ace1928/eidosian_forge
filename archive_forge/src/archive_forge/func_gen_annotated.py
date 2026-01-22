import argparse
import os
import textwrap
from collections import defaultdict
from typing import Any, Dict, List, Sequence
import torchgen.api.python as python
from torchgen.context import with_native_function
from torchgen.gen import parse_native_yaml
from torchgen.model import Argument, BaseOperatorName, NativeFunction
from torchgen.utils import FileManager
from .gen_python_functions import (
def gen_annotated(native_yaml_path: str, tags_yaml_path: str, out: str, autograd_dir: str) -> None:
    native_functions = parse_native_yaml(native_yaml_path, tags_yaml_path).native_functions
    mappings = ((is_py_torch_function, 'torch._C._VariableFunctions'), (is_py_nn_function, 'torch._C._nn'), (is_py_linalg_function, 'torch._C._linalg'), (is_py_special_function, 'torch._C._special'), (is_py_fft_function, 'torch._C._fft'), (is_py_variable_method, 'torch.Tensor'))
    annotated_args: List[str] = []
    for pred, namespace in mappings:
        groups: Dict[BaseOperatorName, List[NativeFunction]] = defaultdict(list)
        for f in native_functions:
            if not should_generate_py_binding(f) or not pred(f):
                continue
            groups[f.func.name.name].append(f)
        for group in groups.values():
            for f in group:
                annotated_args.append(f'{namespace}.{gen_annotated_args(f)}')
    template_path = os.path.join(autograd_dir, 'templates')
    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)
    fm.write_with_template('annotated_fn_args.py', 'annotated_fn_args.py.in', lambda: {'annotated_args': textwrap.indent('\n'.join(annotated_args), '    ')})