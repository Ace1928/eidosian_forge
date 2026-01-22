import re
from .utils import Str, classify, get_regexp_width, Py36, Serialize, suppress
from .exceptions import UnexpectedCharacters, LexError, UnexpectedToken
from copy import copy
@property
def mres(self):
    if self._mres is None:
        self._build()
    return self._mres