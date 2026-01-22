from __future__ import annotations
from abc import abstractmethod
import re
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import FrozenSet
from typing import Iterator
from typing import List
from typing import MutableMapping
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from sqlalchemy.types import NULLTYPE
from . import schemaobj
from .base import BatchOperations
from .base import Operations
from .. import util
from ..util import sqla_compat
@classmethod
def register_add_constraint(cls, type_: str) -> Callable[[Type[_AC]], Type[_AC]]:

    def go(klass: Type[_AC]) -> Type[_AC]:
        cls.add_constraint_ops.dispatch_for(type_)(klass.from_constraint)
        return klass
    return go