import dataclasses
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type
import torch
from torch._export import ExportedProgram
from torch.utils._pytree import (
def register_dataclass_as_pytree_node(cls: Type[Any], flatten_fn: Optional[FlattenFunc]=None, unflatten_fn: Optional[UnflattenFunc]=None, *, serialized_type_name: Optional[str]=None, to_dumpable_context: Optional[ToDumpableContextFn]=None, from_dumpable_context: Optional[FromDumpableContextFn]=None, return_none_fields: bool=False) -> None:
    assert dataclasses.is_dataclass(cls), f'Only dataclasses can be registered with this function: {cls}'
    serialized_type = f'{cls.__module__}.{cls.__qualname__}'
    SERIALIZED_DATACLASS_TO_PYTHON_DATACLASS[serialized_type] = cls

    def default_flatten_fn(obj: Any) -> Tuple[List[Any], Context]:
        flattened = []
        flat_names = []
        none_names = []
        for f in dataclasses.fields(obj):
            name, val = (f.name, getattr(obj, f.name))
            if val is not None or return_none_fields:
                flattened.append(val)
                flat_names.append(name)
            else:
                none_names.append(name)
        return (flattened, (cls, flat_names, none_names))

    def default_unflatten_fn(values: Iterable[Any], context: Context) -> Any:
        typ, flat_names, none_names = context
        return typ(**dict(zip(flat_names, values)), **{k: None for k in none_names})

    def default_to_dumpable_context(context: Context) -> DumpableContext:
        return (serialized_type, context[1], context[2])

    def default_from_dumpable_context(dumpable_context: DumpableContext) -> Context:
        return (SERIALIZED_DATACLASS_TO_PYTHON_DATACLASS[dumpable_context[0]], dumpable_context[1], dumpable_context[2])
    flatten_fn = flatten_fn if flatten_fn is not None else default_flatten_fn
    unflatten_fn = unflatten_fn if unflatten_fn is not None else default_unflatten_fn
    if (to_dumpable_context is None) ^ (from_dumpable_context is None):
        raise ValueError(f'Both to_dumpable_context and from_dumpable_context for {cls} must be None or registered.')
    to_dumpable_context = to_dumpable_context if to_dumpable_context is not None else default_to_dumpable_context
    from_dumpable_context = from_dumpable_context if from_dumpable_context is not None else default_from_dumpable_context
    _register_pytree_node(cls, flatten_fn, unflatten_fn, serialized_type_name=serialized_type_name, to_dumpable_context=to_dumpable_context, from_dumpable_context=from_dumpable_context)