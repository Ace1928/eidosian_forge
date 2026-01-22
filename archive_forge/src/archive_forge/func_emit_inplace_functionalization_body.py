from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union
from torchgen.api import cpp, dispatcher
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.context import (
from torchgen.model import (
from torchgen.native_function_generation import (
from torchgen.selective_build.selector import SelectiveBuilder
@with_native_function_and
def emit_inplace_functionalization_body(f: NativeFunction, g: NativeFunctionsGroup) -> str:
    assert modifies_arguments(f)
    dispatcher_sig = DispatcherSignature.from_schema(f.func)
    unwrap_tensor_args_str, unwrapped_args_ctx = unwrap_tensor_args(dispatcher_sig, is_view_op=False)
    mutated_names = [a.name for a in f.func.arguments.flat_all if a.type.is_tensor_like() and a.annotation is not None]
    non_mutated_names = [a.name for a in f.func.arguments.flat_all if a.type.is_tensor_like() and a.annotation is None]
    non_mutated_tensor_names = [a.name for a in f.func.arguments.flat_all if a.type == BaseType(BaseTy.Tensor) and a.annotation is None]
    check_all_mutated_args_are_functional = ' && '.join(['true'] + [f'at::functionalization::impl::isFunctionalTensor({a})' for a in mutated_names])
    check_any_non_mutated_args_are_functional = ' || '.join(['false'] + [f'at::functionalization::impl::isFunctionalTensor({a})' for a in non_mutated_names])
    check_any_non_mutated_tensors_are_xla = ' || '.join(['false'] + [f'{a}.device().type() == c10::DeviceType::XLA' for a in non_mutated_tensor_names])
    inplace_exprs = [e.expr for e in translate(unwrapped_args_ctx, dispatcher_sig.arguments(), method=False)]
    return_type = dispatcher.returns_type(g.functional.func.returns).remove_const_ref().cpp_type()
    functional_sig = DispatcherSignature.from_schema(g.functional.func)
    functional_exprs = [e.expr for e in translate(unwrapped_args_ctx, functional_sig.arguments(), method=False)]
    if f.func.is_out_fn():
        mutable_input_post_processing = '\n'.join([f'\n      at::functionalization::impl::replace_(\n        {a.name}, {('std::get<' + str(i) + '>(tmp_output)' if len(f.func.returns) > 1 else 'tmp_output')});\n      at::functionalization::impl::commit_update({a.name});' for i, a in enumerate(f.func.arguments.out) if a.annotation and a.annotation.is_write and a.type.is_tensor_like()])
    else:
        mutable_input_post_processing = '\n'.join([f'\n      at::functionalization::impl::replace_({a.name}, tmp_output);\n      at::functionalization::impl::commit_update({a.name});' for a in f.func.arguments.flat_all if a.annotation and a.annotation.is_write and a.type.is_tensor_like()])
    meta_conversion_str, meta_call_ctx = convert_to_meta_tensors(dispatcher_sig)
    any_storage_args = any((a.type == BaseType(BaseTy.Storage) for a in f.func.arguments.flat_all))
    return f"""\n    {dispatcher_sig.defn(name=wrapper_name(f.func), is_redispatching_fn=True)} {{\n      if ({str(not any_storage_args and f.func.kind() == SchemaKind.inplace).lower()}) {{\n        // Before converting the mutable op to its functional variant, run meta tensors through the original op.\n        // This will help us catch shape errors that apply to inplace ops that wouldn't apply to their functional variants.\n        // (We can only do this for inplace ops today though, because they technically all support meta tensors).\n        {meta_conversion_str}\n        at::AutoDispatchSkipFunctionalize func_guard;\n        c10::impl::ExcludeDispatchKeyGuard guard(exclude_keys_for_meta_dispatch);\n        at::_ops::{f.func.name.unambiguous_name()}::call({', '.join((a.name for a in meta_call_ctx))});\n      }}\n      {unwrap_tensor_args_str}\n      if (!({check_all_mutated_args_are_functional})) {{\n        // We want to disable this check if there are any XLA tensors.\n        // cpu_tensor.copy_(xla_tensor) is valid code.\n        if (!({check_any_non_mutated_tensors_are_xla}) && ({check_any_non_mutated_args_are_functional})) {{\n         // case 1: trying to mutate a non functional tensor with a functional tensor is an error\n         TORCH_INTERNAL_ASSERT(false,\n           "mutating a non-functional tensor with a functional tensor is not allowed.",\n           " Please ensure that all of your inputs are wrapped inside of a functionalize() call.");\n        }} else {{\n         // case 2: arguments are not functional tensors, so we no-op and redispatch.\n         at::AutoDispatchSkipFunctionalize guard;\n         {maybe_create_output(f, 'tmp_output')}at::_ops::{f.func.name.unambiguous_name()}::call({', '.join(inplace_exprs)});\n         {return_from_mutable_noop_redispatch(f, 'tmp_output')};\n        }}\n      }} else {{\n        {return_type} tmp_output;\n        {{\n          at::AutoDispatchSkipFunctionalize guard;\n          tmp_output = at::_ops::{g.functional.func.name.unambiguous_name()}::call({', '.join(functional_exprs)});\n        }}\n        {wrap_propagate_mutations_and_return(f, g.functional, 'tmp_output')}\n      }}\n    }}"""