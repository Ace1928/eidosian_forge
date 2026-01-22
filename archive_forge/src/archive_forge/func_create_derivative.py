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
def create_derivative(f: NativeFunction, formula: str, var_names: Tuple[str, ...], available_named_gradients: Sequence[str]) -> Derivative:
    original_formula = formula
    arguments: List[NamedCType] = [a.nctype.remove_const_ref() for a in cpp_arguments(f)]
    return_names = tuple((n if n != 'self' else 'result' for n in cpp.return_names(f)))
    return_types = tuple((cpp.return_type(r, symint=True).remove_const_ref() for r in f.func.returns))
    named_returns = [NamedCType(name, type) for name, type in zip(return_names, return_types)]
    formula, saved_inputs = saved_variables(formula, arguments, var_names)
    formula, saved_outputs = saved_variables(formula, named_returns, var_names)
    used_named_gradients = {name for name in available_named_gradients if re.search(IDENT_REGEX.format(name), formula)}
    for i in used_gradient_indices(formula):
        if i >= len(f.func.returns):
            raise RuntimeError(f'Out of bounds grads access: derivative formula for {cpp.name(f.func)} used grads[{i}], but the forward only returns {len(f.func.returns)} outputs.')
    return Derivative(formula=formula, original_formula=original_formula, var_names=var_names, saved_inputs=saved_inputs, saved_outputs=saved_outputs, named_gradients=used_named_gradients)