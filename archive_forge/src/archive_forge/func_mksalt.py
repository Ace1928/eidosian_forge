import sys as _sys
import errno
import string as _string
import warnings
from random import SystemRandom as _SystemRandom
from collections import namedtuple as _namedtuple
def mksalt(method=None, *, rounds=None):
    """Generate a salt for the specified method.

    If not specified, the strongest available method will be used.

    """
    if method is None:
        method = methods[0]
    if rounds is not None and (not isinstance(rounds, int)):
        raise TypeError(f'{rounds.__class__.__name__} object cannot be interpreted as an integer')
    if not method.ident:
        s = ''
    else:
        s = f'${method.ident}$'
    if method.ident and method.ident[0] == '2':
        if rounds is None:
            log_rounds = 12
        else:
            log_rounds = int.bit_length(rounds - 1)
            if rounds != 1 << log_rounds:
                raise ValueError('rounds must be a power of 2')
            if not 4 <= log_rounds <= 31:
                raise ValueError('rounds out of the range 2**4 to 2**31')
        s += f'{log_rounds:02d}$'
    elif method.ident in ('5', '6'):
        if rounds is not None:
            if not 1000 <= rounds <= 999999999:
                raise ValueError('rounds out of the range 1000 to 999_999_999')
            s += f'rounds={rounds}$'
    elif rounds is not None:
        raise ValueError(f"{method} doesn't support the rounds argument")
    s += ''.join((_sr.choice(_saltchars) for char in range(method.salt_chars)))
    return s