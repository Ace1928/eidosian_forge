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
def node_base_ctor_call(self, schema: LazyIrSchema) -> str:
    value_args = schema.filtered_args(values=True, scalars=False)
    base_ctor_value_args_list = []
    for arg in value_args:
        if isinstance(arg.lazy_type, (BaseCType, VectorCType)):
            base_ctor_value_args_list.append(f'{arg.name}')
        elif isinstance(arg.lazy_type, OptionalCType):
            base_ctor_value_args_list.append(f'{arg.name}.value_or(kNullValue)')
        else:
            raise AssertionError(f'Unsupported type ({arg.lazy_type}) - add support if necessary')
    base_ctor_value_args = ', '.join(base_ctor_value_args_list)
    scalar_args = schema.filtered_args(values=False, scalars=True)
    if schema.properties.ShapePrecompute:
        shape_ctor_arg = 'std::move(shapes),'
    elif schema.properties.ShapeCompute:
        shape_args = [a.name for a in value_args]
        shape_args.extend((a.name for a in scalar_args))
        shape_ctor_arg = f'compute_shape_{schema.name}({', '.join(shape_args)}),'
    elif schema.properties.ShapeCache:
        shape_args = [f'operand({i})' for i in range(len(value_args))]
        shape_args.extend((a.name for a in scalar_args))
        shape_ctor_arg = f'[&](){{ return compute_shape_{schema.name}({', '.join(shape_args)})[0]; }},'
    else:
        shape_ctor_arg = ''
    scalar_hashes = ', '.join((f'{a.name}' for a in scalar_args))
    return f'{self.node_base}(\n              {schema.node_name}::ClassOpKind(),\n              OpList{{{base_ctor_value_args}}},\n              {shape_ctor_arg}\n              /* num_outputs */ {len(schema.returns)},\n              torch::lazy::MHash({scalar_hashes}))'