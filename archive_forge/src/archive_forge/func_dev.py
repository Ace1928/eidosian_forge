import collections
import itertools
import re
from typing import Any, Callable, Optional, SupportsInt, Tuple, Union
from ._structures import Infinity, InfinityType, NegativeInfinity, NegativeInfinityType
@property
def dev(self) -> Optional[int]:
    """The development number of the version.

        >>> print(Version("1.2.3").dev)
        None
        >>> Version("1.2.3.dev1").dev
        1
        """
    return self._version.dev[1] if self._version.dev else None