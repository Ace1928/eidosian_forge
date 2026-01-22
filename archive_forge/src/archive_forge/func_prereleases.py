import abc
import itertools
import re
from typing import Callable, Iterable, Iterator, List, Optional, Tuple, TypeVar, Union
from .utils import canonicalize_version
from .version import Version
@prereleases.setter
def prereleases(self, value: bool) -> None:
    self._prereleases = value