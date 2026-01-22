import re
from numba.core import types
def mangle_type_or_value(typ):
    """
    Mangle type parameter and arbitrary value.
    """
    if isinstance(typ, types.Type):
        if typ in N2CODE:
            return N2CODE[typ]
        else:
            return mangle_templated_ident(*typ.mangling_args)
    elif isinstance(typ, int):
        return 'Li%dE' % typ
    elif isinstance(typ, str):
        return mangle_identifier(typ)
    else:
        enc = _escape_string(str(typ))
        return _len_encoded(enc)