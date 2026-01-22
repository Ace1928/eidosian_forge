from numba.core.types.abstract import Callable, Literal, Type, Hashable
from numba.core.types.common import (Dummy, IterableType, Opaque,
from numba.core.typeconv import Conversion
from numba.core.errors import TypingError, LiteralTypingError
from numba.core.ir import UndefinedType
from numba.core.utils import get_hashable_key
def unliteral(lit_type):
    """
    Get base type from Literal type.
    """
    if hasattr(lit_type, '__unliteral__'):
        return lit_type.__unliteral__()
    return getattr(lit_type, 'literal_type', lit_type)