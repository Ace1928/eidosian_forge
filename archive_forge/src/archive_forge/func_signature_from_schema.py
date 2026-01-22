from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
from torchgen.api import cpp
from torchgen.api.types import Binding, CppSignature, CppSignatureGroup
from torchgen.gen import pythonify_default
from torchgen.model import (
def signature_from_schema(func: FunctionSchema, *, category_override: Optional[str], method: bool=False, pyi: bool=False) -> PythonSignature:
    args: List[Argument] = []
    args.extend(func.arguments.pre_self_positional)
    if not method and func.arguments.self_arg is not None:
        args.append(func.arguments.self_arg.argument)
    args.extend(func.arguments.post_self_positional)
    args.extend(func.arguments.pre_tensor_options_kwarg_only)
    args.extend(func.arguments.post_tensor_options_kwarg_only)
    args.extend(func.arguments.out)
    input_arg_set = {a.name for a in func.arguments.flat_positional}
    kwarg_only_set = {a.name for a in func.arguments.flat_kwarg_only}
    out_arg_set = {a.name for a in func.arguments.out}
    input_args = tuple(map(argument, filter(lambda a: a.name in input_arg_set, args)))
    input_kwargs = tuple(map(argument, filter(lambda a: a.name in kwarg_only_set, args)))
    outputs = tuple(map(argument, filter(lambda a: a.name in out_arg_set, args)))
    has_tensor_input_arg = any((a.type.is_tensor_like() for a in func.arguments.flat_non_out))
    if any((a.name == 'requires_grad' for a in func.schema_order_arguments())):
        raise ValueError('argument named requires_grad is reserved, should not explicitly add it in the schema')
    has_tensor_return = any((r.type.is_tensor_like() for r in func.returns))
    name: str = cpp.name(func)
    is_factory_function = category_override == 'factory' or (has_tensor_return and (not has_tensor_input_arg))
    is_like_or_new_function = category_override in ('new', 'like') or name.startswith('new_') or name.endswith('_like')
    tensor_options_args: List[PythonArgument] = []
    if is_factory_function or is_like_or_new_function:

        def topt_default_init(name: str) -> Optional[str]:
            topt_args = func.arguments.tensor_options
            if topt_args is None:
                return None
            a = getattr(topt_args, name)
            if a.default is None or a.default == 'None':
                return None
            return cpp.default_expr(a.default, a.type, symint=False)
        tensor_options_args.append(PythonArgument(name='dtype', type=OptionalType(BaseType(BaseTy.ScalarType)), default='None', default_init=None if is_like_or_new_function else topt_default_init('dtype')))
        tensor_options_args.append(PythonArgument(name='layout', type=OptionalType(BaseType(BaseTy.Layout)), default='None', default_init=None if is_like_or_new_function else topt_default_init('layout')))
        tensor_options_args.append(PythonArgument(name='device', type=OptionalType(BaseType(BaseTy.Device)), default='None', default_init=None if is_like_or_new_function else topt_default_init('device') or 'torch::tensors::get_default_device()'))
        tensor_options_args.append(PythonArgument(name='pin_memory', type=OptionalType(BaseType(BaseTy.bool)), default='False', default_init=None))
        tensor_options_args.append(PythonArgument(name='requires_grad', type=OptionalType(BaseType(BaseTy.bool)), default='False', default_init=None))
    returns = PythonReturns(returns=func.returns)
    return PythonSignature(name=str(func.name.name), input_args=input_args, input_kwargs=input_kwargs, output_args=PythonOutArgument.from_outputs(outputs), tensor_options_args=tuple(tensor_options_args), returns=returns, method=method)