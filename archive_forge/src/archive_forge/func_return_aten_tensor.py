import itertools
from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import torchgen.api.dispatcher as dispatcher
from torchgen.api.lazy import (
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.context import method_with_native_function
from torchgen.dest.lazy_ts_lowering import ts_lowering_body
from torchgen.model import (
def return_aten_tensor(self, func: NativeFunction, schema: LazyIrSchema) -> str:
    returns_length = len(schema.returns)
    value_args = schema.filtered_args(values=True, scalars=False)
    value_types_names = [f'{a.name}' for a in value_args if not a.is_wrapped_scalar]
    first_tensor_name = value_types_names[0] if len(value_types_names) > 0 else None
    bridge_str = f'auto result = {self.create_aten_from_ltc_tensor}(\n                {self.create_lazy_tensor(first_tensor_name)}(std::move(node), *common_device));'
    if returns_length > 1:
        assert len(value_types_names) > 0, 'Code below assumes there is at least one tensor arg'
        bridge_str = f'std::vector<{self.lazy_tensor_ptr}> lazy_tensors;\n        for (int i = 0; i < {returns_length}; i++) {{\n            lazy_tensors.push_back({self.create_lazy_tensor(first_tensor_name)}({getValueT()}(node, i), *common_device));\n        }}\n        auto result = {self.tuple_aten_from_ltc_tensors}<{returns_length}>(lazy_tensors);'
    if schema.name.name.inplace or func.func.is_out_fn():
        assert returns_length == 1, f'We assumed there was no such case where an op is an in-place variant and has tuple outputs, but got tuple of len {returns_length}.'
        bridge_str = f'lazy_{first_tensor_name}->SetInPlaceIrValue(node);\n        auto& result = {first_tensor_name};'
    bridge_str += '\n        return result;'
    return bridge_str