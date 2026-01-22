import datetime
import math
import typing as t
from wandb.util import (
@staticmethod
def type_from_dtype(dtype: ConvertableToType) -> 'Type':
    if isinstance(dtype, Type):
        wbtype: Type = dtype
    elif isinstance(dtype, type) and issubclass(dtype, Type):
        wbtype = dtype()
    elif isinstance(dtype, type):
        handler = TypeRegistry.types_by_class().get(dtype)
        if handler:
            wbtype = handler()
        else:
            wbtype = PythonObjectType.from_obj(dtype)
    elif isinstance(dtype, list):
        if len(dtype) == 0:
            wbtype = ListType()
        elif len(dtype) == 1:
            wbtype = ListType(TypeRegistry.type_from_dtype(dtype[0]))
        else:
            wbtype = UnionType([TypeRegistry.type_from_dtype(dt) for dt in dtype])
    elif isinstance(dtype, dict):
        wbtype = TypedDictType({key: TypeRegistry.type_from_dtype(dtype[key]) for key in dtype})
    else:
        wbtype = ConstType(dtype)
    return wbtype