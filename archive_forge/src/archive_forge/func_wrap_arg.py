import abc
from numba.core.typing.typeof import typeof, Purpose
def wrap_arg(value, default=InOut):
    return value if isinstance(value, ArgHint) else default(value)