import textwrap
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
from torchgen.api.translate import translate
from torchgen.api.types import DispatcherSignature
from torchgen.context import method_with_native_function
from torchgen.model import (
from torchgen.utils import mapMaybe
def gen_case_where_all_bdims_are_none(outer_sig: DispatcherSignature, schema: FunctionSchema, cur_level_var: str) -> str:
    conditions = []
    flat_args = schema.arguments.flat_all
    for arg in flat_args:
        if not arg.type.is_tensor_like():
            continue
        conditions.append(f'!isBatchedAtLevel({arg.name}, {cur_level_var})')
    sig = DispatcherSignature.from_schema(schema)
    translated_args = ', '.join((e.expr for e in translate(outer_sig.arguments(), sig.arguments())))
    return f'if ({' && '.join(conditions)}) {{\n  return at::_ops::{sig.func.name.unambiguous_name()}::call({translated_args});\n}}'