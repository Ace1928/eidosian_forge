from numba.core import types, typing

    From *sig* (a signature specification), return a ``(args, return_type)``
    tuple, where ``args`` itself is a tuple of types, and ``return_type``
    can be None if not specified.
    