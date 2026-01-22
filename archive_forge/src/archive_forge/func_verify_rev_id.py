from __future__ import annotations
import collections
import re
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Deque
from typing import Dict
from typing import FrozenSet
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import overload
from typing import Protocol
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from sqlalchemy import util as sqlautil
from .. import util
from ..util import not_none
@classmethod
def verify_rev_id(cls, revision: str) -> None:
    illegal_chars = set(revision).intersection(_revision_illegal_chars)
    if illegal_chars:
        raise RevisionError("Character(s) '%s' not allowed in revision identifier '%s'" % (', '.join(sorted(illegal_chars)), revision))