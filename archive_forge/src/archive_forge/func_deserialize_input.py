import base64
import dataclasses
import io
import json
import logging
import math
import operator
import typing
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Union
import sympy
import torch
import torch.export.exported_program as ep
from torch._export.verifier import load_verifier
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx.experimental import symbolic_shapes
from torch.utils._pytree import treespec_dumps, treespec_loads
from torch.utils._sympy.value_ranges import ValueRanges
from .schema import (  # type: ignore[attr-defined]
from torch.export.exported_program import (
from .upgrade import GraphModuleOpUpgrader
def deserialize_input(self, inp: Argument) -> Any:
    value = inp.value
    typ_ = inp.type
    if typ_ == 'as_none':
        return None
    elif typ_ == 'as_scalar_type':
        return _SERIALIZE_TO_TORCH_DTYPE[value]
    elif typ_ == 'as_memory_format':
        return _SERIALIZE_TO_TORCH_MEMORY_FORMAT[value]
    elif typ_ == 'as_layout':
        return _SERIALIZE_TO_TORCH_LAYOUT[value]
    elif typ_ == 'as_graph':
        assert isinstance(value, GraphArgument)
        with self.save_graph_module():
            self.deserialize_graph(value.graph)
            submodule = torch._export.exported_program._create_graph_module_for_export(self.module, self.graph)
        self.module.register_module(value.name, submodule)
        return self.graph.create_node('get_attr', value.name, name=value.name)
    elif isinstance(value, Device):
        return deserialize_device(value)
    elif isinstance(value, TensorArgument):
        return self.serialized_name_to_node[value.name]
    elif isinstance(value, (int, float, bool)):
        return value
    elif isinstance(value, str):
        return str(value)
    elif isinstance(value, (SymIntArgument, SymBoolArgument)):
        return self.deserialize_sym_argument(value)
    elif isinstance(value, list):
        if len(value) == 0:
            return []
        elif isinstance(value[0], TensorArgument):
            result = []
            for arg in value:
                result.append(self.serialized_name_to_node[arg.name])
            return result
        elif isinstance(value[0], (int, float, bool)):
            return list(value)
        elif isinstance(value[0], (SymIntArgument, SymBoolArgument)):
            return [self.deserialize_sym_argument(arg) for arg in value]
        elif isinstance(value[0], OptionalTensorArgument):

            def deserialize_optional_tensor_args(a):
                if a.type == 'as_none':
                    return None
                elif a.type == 'as_tensor':
                    return self.serialized_name_to_node[a.value]
                else:
                    raise SerializeError(f'Unhandled argument {inp}')
            return list(map(deserialize_optional_tensor_args, value))
        else:
            raise SerializeError(f'Unhandled argument {inp}')
    elif isinstance(value, CustomObjArgument):
        return self.constants[value.name]
    else:
        raise SerializeError(f'Unhandled argument {inp}')