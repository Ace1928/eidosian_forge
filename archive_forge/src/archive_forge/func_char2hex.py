import json
from ..error import GraphQLSyntaxError
def char2hex(a):
    """Converts a hex character to its integer value.
    '0' becomes 0, '9' becomes 9
    'A' becomes 10, 'F' becomes 15
    'a' becomes 10, 'f' becomes 15

    Returns -1 on error."""
    if 48 <= a <= 57:
        return a - 48
    elif 65 <= a <= 70:
        return a - 55
    elif 97 <= a <= 102:
        return a - 87
    return -1