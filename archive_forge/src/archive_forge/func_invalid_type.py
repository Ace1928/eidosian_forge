from abc import ABCMeta
from typing import Any, Dict, Optional, Pattern, Tuple, Type
import re
def invalid_type(cls: Type[Any], value: object) -> TypeError:
    if cls.name:
        return TypeError('invalid type {!r} for xs:{}'.format(type(value), cls.name))
    return TypeError('invalid type {!r} for {!r}'.format(type(value), cls))