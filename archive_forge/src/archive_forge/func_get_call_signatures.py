from numba.core.types.abstract import Callable, Literal, Type, Hashable
from numba.core.types.common import (Dummy, IterableType, Opaque,
from numba.core.typeconv import Conversion
from numba.core.errors import TypingError, LiteralTypingError
from numba.core.ir import UndefinedType
from numba.core.utils import get_hashable_key
def get_call_signatures(self):
    if not self.cm.is_callable:
        msg = 'contextmanager {} is not callable'.format(self.cm)
        raise TypingError(msg)
    return ((), False)