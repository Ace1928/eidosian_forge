import collections
from typing import Any, Set
import weakref
@property
def unwrapped(self):
    return self._wrapped()