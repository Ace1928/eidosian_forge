import functools
import string
import re
from types import MappingProxyType
from llvmlite.ir import values, types, _utils
from llvmlite.ir._utils import (_StrCaching, _StringReferenceCaching,
@classmethod
def literal_struct(cls, elems):
    """
        Construct a literal structure constant made of the given members.
        """
    tys = [el.type for el in elems]
    return cls(types.LiteralStructType(tys), elems)