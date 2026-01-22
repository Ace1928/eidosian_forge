import sys
from copy import deepcopy
from typing import List, Callable, Iterator, Union, Optional, Generic, TypeVar, TYPE_CHECKING
from collections import OrderedDict
@property
def meta(self) -> Meta:
    if self._meta is None:
        self._meta = Meta()
    return self._meta