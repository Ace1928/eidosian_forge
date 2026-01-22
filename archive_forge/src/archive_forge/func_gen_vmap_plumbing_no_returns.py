import textwrap
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
from torchgen.api.translate import translate
from torchgen.api.types import DispatcherSignature
from torchgen.context import method_with_native_function
from torchgen.model import (
from torchgen.utils import mapMaybe
def gen_vmap_plumbing_no_returns(native_function: NativeFunction) -> str:
    schema = native_function.func
    sig = DispatcherSignature.from_schema(schema)
    cur_level_var = 'cur_level'
    unwraps, unwrapped_arg_list = gen_unwraps(schema.arguments.flat_all, cur_level_var)
    bdims_all_none_case = gen_case_where_all_bdims_are_none(sig, schema, cur_level_var)
    return f'template <typename batch_rule_t, batch_rule_t batch_rule>\n{sig.decl(name=schema.name.unambiguous_name() + '_generated_plumbing')} {{\n  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);\n  auto maybe_layer = maybeCurrentDynamicLayer();\n  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing_no_returns");\n  int64_t {cur_level_var} = maybe_layer->layerId();\n{textwrap.indent(bdims_all_none_case, '  ')}\n{textwrap.indent(unwraps, '  ')}\n  batch_rule({', '.join(unwrapped_arg_list)});\n}}'