from collections import UserDict
from collections.abc import Mapping
def strencode(instr, encoding='utf-8'):
    try:
        instr = instr.encode(encoding)
    except (UnicodeDecodeError, AttributeError):
        pass
    return instr