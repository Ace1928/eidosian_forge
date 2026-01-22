import logging
import re
from typing import (
from . import settings
from .utils import choplist
def literal_name(x: object) -> Any:
    if not isinstance(x, PSLiteral):
        if settings.STRICT:
            raise PSTypeError('Literal required: {!r}'.format(x))
        else:
            name = x
    else:
        name = x.name
        if not isinstance(name, str):
            try:
                name = str(name, 'utf-8')
            except Exception:
                pass
    return name