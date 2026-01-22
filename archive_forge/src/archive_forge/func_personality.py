import functools
import string
import re
from types import MappingProxyType
from llvmlite.ir import values, types, _utils
from llvmlite.ir._utils import (_StrCaching, _StringReferenceCaching,
@personality.setter
def personality(self, val):
    assert val is None or isinstance(val, GlobalValue)
    self._personality = val