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
@upgrade_ops.setter
def upgrade_ops(self, upgrade_ops: Union[UpgradeOps, List[UpgradeOps]]) -> None:
    self._upgrade_ops = util.to_list(upgrade_ops)
    for elem in self._upgrade_ops:
        assert isinstance(elem, UpgradeOps)