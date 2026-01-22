import textwrap
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
from torchgen.api.translate import translate
from torchgen.api.types import DispatcherSignature
from torchgen.context import method_with_native_function
from torchgen.model import (
from torchgen.utils import mapMaybe
def gen_vmap_inplace_plumbing(native_function: NativeFunction) -> Optional[str]:
    schema = native_function.func
    sig = DispatcherSignature.from_schema(schema)
    returns = schema.returns
    assert schema.kind() == SchemaKind.inplace
    if not is_mutated_arg(schema.arguments.flat_all[0]):
        return None
    if not len([arg for arg in schema.arguments.flat_all if is_mutated_arg(arg)]) == 1:
        return None
    if len(returns) == 0:
        return None
    if not all((is_tensor(ret.type) or is_tensor_list(ret.type) for ret in returns)):
        return None
    if not accepts_at_least_one_tensor_input(schema):
        return None
    cur_level_var = 'cur_level'
    unwraps, unwrapped_arg_list = gen_unwraps(schema.arguments.flat_all, cur_level_var)
    bdims_all_none_case = gen_case_where_all_bdims_are_none(sig, schema, cur_level_var)
    return f'template <typename batch_rule_t, batch_rule_t batch_rule>\n{sig.decl(name=schema.name.unambiguous_name() + '_generated_plumbing')} {{\n  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);\n  auto maybe_layer = maybeCurrentDynamicLayer();\n  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");\n  int64_t {cur_level_var} = maybe_layer->layerId();\n{textwrap.indent(bdims_all_none_case, '  ')}\n{textwrap.indent(unwraps, '  ')}\n  batch_rule({', '.join(unwrapped_arg_list)});\n  return {schema.arguments.flat_all[0].name};\n}}'