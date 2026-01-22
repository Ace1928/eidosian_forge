import collections
import itertools
import re
from typing import Any, Callable, Optional, SupportsInt, Tuple, Union
from ._structures import Infinity, InfinityType, NegativeInfinity, NegativeInfinityType
@property
def base_version(self) -> str:
    """The "base version" of the version.

        >>> Version("1.2.3").base_version
        '1.2.3'
        >>> Version("1.2.3+abc").base_version
        '1.2.3'
        >>> Version("1!1.2.3+abc.dev1").base_version
        '1!1.2.3'

        The "base version" is the public version of the project without any pre or post
        release markers.
        """
    parts = []
    if self.epoch != 0:
        parts.append(f'{self.epoch}!')
    parts.append('.'.join((str(x) for x in self.release)))
    return ''.join(parts)