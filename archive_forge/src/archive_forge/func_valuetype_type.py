from typing import List, Optional, Sequence, Set, Union
from torchgen import local
from torchgen.api.types import (
from torchgen.model import (
from torchgen.utils import assert_never
from .types import (
def valuetype_type(t: Type, *, binds: ArgName, remove_non_owning_ref_types: bool=False) -> Optional[NamedCType]:
    if isinstance(t, BaseType):
        if t.name == BaseTy.Tensor or t.name == BaseTy.Scalar:
            return None
        elif str(t) == 'SymInt':
            return NamedCType(binds, BaseCType(BaseTypeToCppMapping[BaseTy.int]))
        if remove_non_owning_ref_types:
            if t.name == BaseTy.str:
                raise AssertionError('string ref->value conversion: not implemented yet')
        return NamedCType(binds, BaseCType(BaseTypeToCppMapping[t.name]))
    elif isinstance(t, OptionalType):
        elem = valuetype_type(t.elem, binds=binds)
        if elem is None:
            return None
        return NamedCType(binds, OptionalCType(elem.type))
    elif isinstance(t, ListType):
        if str(t.elem) == 'bool':
            assert t.size is not None
            return NamedCType(binds, ArrayCType(BaseCType(BaseTypeToCppMapping[BaseTy.bool]), t.size))
        else:
            return None
    else:
        raise AssertionError(f'unrecognized type {repr(t)}')