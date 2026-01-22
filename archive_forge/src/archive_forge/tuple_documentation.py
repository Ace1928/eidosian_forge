from numba.core import types, typing, errors
from numba.core.cgutils import alloca_once
from numba.core.extending import intrinsic
This exists to handle the situation y = (*x,), the interpreter injects a
    call to it in the case of a single value unpack. It's not possible at
    interpreting time to differentiate between an unpack on a variable sized
    container e.g. list and a fixed one, e.g. tuple. This function handles the
    situation should it arise.
    