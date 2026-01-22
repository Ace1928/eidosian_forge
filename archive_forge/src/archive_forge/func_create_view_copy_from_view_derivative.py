import re
from dataclasses import dataclass
from typing import cast, Dict, List, Match, Optional, Sequence, Set, Tuple
from torchgen import local
from torchgen.api import cpp
from torchgen.api.types import BaseCType, Binding, NamedCType, tensorListT
from torchgen.model import (
from torchgen.utils import IDENT_REGEX
def create_view_copy_from_view_derivative(self, g: NativeFunctionsViewGroup) -> Optional['DifferentiabilityInfo']:
    if g.view_copy is None:
        return None
    f = g.view_copy
    name_split_by_period = self.name.split('.', maxsplit=2)
    view_copy_name = f'{name_split_by_period[0]}_copy.' + '.'.join(name_split_by_period[1:])
    view_copy_op_name = None if self.op is None else f'{self.op}_copy'
    return DifferentiabilityInfo(name=view_copy_name, func=f, op=view_copy_op_name, derivatives=self.derivatives, forward_derivatives=self.forward_derivatives, all_saved_inputs=self.all_saved_inputs, all_saved_outputs=self.all_saved_outputs, available_named_gradients=self.available_named_gradients, used_named_gradients=self.used_named_gradients, args_with_derivatives=self.args_with_derivatives, non_differentiable_arg_names=self.non_differentiable_arg_names, output_differentiability=self.output_differentiability, output_differentiability_conditions=self.output_differentiability_conditions)