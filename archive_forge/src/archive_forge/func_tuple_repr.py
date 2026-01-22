import re
def tuple_repr(x, unknown_handler=str):
    return '(' + ','.join((name_repr(_, unknown_handler) for _ in x)) + (',)' if len(x) == 1 else ')')