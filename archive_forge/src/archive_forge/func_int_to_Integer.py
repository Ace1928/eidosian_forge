from sympy.external.gmpy import GROUND_TYPES
from sympy.external.importtools import version_tuple
from sympy.interactive.printing import init_printing
from sympy.utilities.misc import ARCH
from sympy import *
def int_to_Integer(s):
    """
    Wrap integer literals with Integer.

    This is based on the decistmt example from
    https://docs.python.org/3/library/tokenize.html.

    Only integer literals are converted.  Float literals are left alone.

    Examples
    ========

    >>> from sympy import Integer # noqa: F401
    >>> from sympy.interactive.session import int_to_Integer
    >>> s = '1.2 + 1/2 - 0x12 + a1'
    >>> int_to_Integer(s)
    '1.2 +Integer (1 )/Integer (2 )-Integer (0x12 )+a1 '
    >>> s = 'print (1/2)'
    >>> int_to_Integer(s)
    'print (Integer (1 )/Integer (2 ))'
    >>> exec(s)
    0.5
    >>> exec(int_to_Integer(s))
    1/2
    """
    from tokenize import generate_tokens, untokenize, NUMBER, NAME, OP
    from io import StringIO

    def _is_int(num):
        """
        Returns true if string value num (with token NUMBER) represents an integer.
        """
        if '.' in num or 'j' in num.lower() or 'e' in num.lower():
            return False
        return True
    result = []
    g = generate_tokens(StringIO(s).readline)
    for toknum, tokval, _, _, _ in g:
        if toknum == NUMBER and _is_int(tokval):
            result.extend([(NAME, 'Integer'), (OP, '('), (NUMBER, tokval), (OP, ')')])
        else:
            result.append((toknum, tokval))
    return untokenize(result)