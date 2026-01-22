import argparse
import os
from typing import List
from torchgen.api import cpp
from torchgen.api.autograd import (
from torchgen.gen import parse_native_yaml
from torchgen.selective_build.selector import SelectiveBuilder
from . import gen_python_functions
from .gen_autograd_functions import (
from .gen_inplace_or_view_type import gen_inplace_or_view_type
from .gen_trace_type import gen_trace_type
from .gen_variable_factories import gen_variable_factories
from .gen_variable_type import gen_variable_type
from .load_derivatives import load_derivatives
def gen_autograd(native_functions_path: str, tags_path: str, out: str, autograd_dir: str, operator_selector: SelectiveBuilder, disable_autograd: bool=False) -> None:
    differentiability_infos, used_dispatch_keys = load_derivatives(os.path.join(autograd_dir, 'derivatives.yaml'), native_functions_path, tags_path)
    template_path = os.path.join(autograd_dir, 'templates')
    native_funcs = parse_native_yaml(native_functions_path, tags_path).native_functions
    fns = sorted(filter(operator_selector.is_native_function_selected_for_training, native_funcs), key=lambda f: cpp.name(f.func))
    fns_with_diff_infos: List[NativeFunctionWithDifferentiabilityInfo] = match_differentiability_info(fns, differentiability_infos)
    if not disable_autograd:
        gen_variable_type(out, native_functions_path, tags_path, fns_with_diff_infos, template_path, used_dispatch_keys)
        gen_inplace_or_view_type(out, native_functions_path, tags_path, fns_with_diff_infos, template_path)
        gen_trace_type(out, native_funcs, template_path)
    gen_autograd_functions_lib(out, differentiability_infos, template_path)
    gen_variable_factories(out, native_functions_path, tags_path, template_path)