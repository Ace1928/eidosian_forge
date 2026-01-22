from typing import Any, Dict, List, Optional, Tuple, Union
from torchgen.api.types import (
from torchgen.model import (
def isValueType(typ: CType, properties: 'Optional[LazyIrProperties]'=None) -> bool:
    """
    Given a type, determine if it is a Value-like type.  This is equivalent to
    being Tensor-like, but assumes the type has already been transformed.
    """
    if isinstance(typ, BaseCType):
        treat_scalars_as_constants = properties and properties.TreatScalarsAsConstants
        return typ.type == getValueT() or (typ.type == scalarT and (not treat_scalars_as_constants)) or typ.type == SymIntT
    elif typ == VectorCType(BaseCType(SymIntT)):
        return False
    elif isinstance(typ, (OptionalCType, ListCType, VectorCType)):
        return isValueType(typ.elem, properties)
    return False