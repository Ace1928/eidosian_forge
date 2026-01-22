from numba.core.types.abstract import Callable, Literal, Type, Hashable
from numba.core.types.common import (Dummy, IterableType, Opaque,
from numba.core.typeconv import Conversion
from numba.core.errors import TypingError, LiteralTypingError
from numba.core.ir import UndefinedType
from numba.core.utils import get_hashable_key
def can_convert_from(self, typingctx, other):
    if isinstance(other, NoneType):
        return Conversion.promote
    elif isinstance(other, Optional):
        return typingctx.can_convert(other.type, self.type)
    else:
        conv = typingctx.can_convert(other, self.type)
        if conv is not None:
            return max(conv, Conversion.promote)