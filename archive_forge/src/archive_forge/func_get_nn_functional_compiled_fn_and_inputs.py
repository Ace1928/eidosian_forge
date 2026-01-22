from torch.jit.annotations import BroadcastingList2, BroadcastingList3  # noqa: F401
import torch.nn.functional as F
import torch
import torch.cuda
import torch.jit
import torch.jit._logging
import torch.jit.frontend
from torch.testing._internal.common_nn import module_tests, new_module_tests
from torch.testing._internal.common_utils import is_iterable_of_tensors
import collections
from copy import deepcopy
from typing import Any, Dict, List, Union
import math  # noqa: F401
from torch import inf
def get_nn_functional_compiled_fn_and_inputs(name, self_size, args, variant_name='', *extra_args):
    test_name = 'test_nn_' + name
    if variant_name != '':
        test_name = test_name + '_' + variant_name
    no_grad = variant_name == 'inplace'
    self_variable = create_input((self_size,))[0][0]
    kwargs = None
    args_variable, kwargs_variable = create_input(args)
    self_tensor = deepcopy(self_variable.data)
    args_tensor = deepcopy(unpack_variables(args_variable))
    f_args_variable = (self_variable,) + args_variable
    f_args_tensor = (self_tensor,) + args_tensor
    with torch._jit_internal._disable_emit_hooks():
        script_fn, inputs = gen_script_fn_and_args(name, 'nn_functional', *f_args_variable)
    return (script_fn, inputs)