from typing import Any, Iterable
from hamcrest.core.description import Description
from hamcrest.core.helpers.hasmethod import hasmethod
from hamcrest.core.helpers.ismock import ismock
def character_in_python_syntax(ch: str) -> str:
    if ch == "'":
        return "'"
    elif ch == '\n':
        return '\\n'
    elif ch == '\r':
        return '\\r'
    elif ch == '\t':
        return '\\t'
    else:
        return ch