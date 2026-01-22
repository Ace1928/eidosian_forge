from __future__ import annotations
from enum import Enum
import operator
import typing
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generic
from typing import no_type_check
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import exc
from ._typing import insp_is_mapper
from .. import exc as sa_exc
from .. import inspection
from .. import util
from ..sql import roles
from ..sql.elements import SQLColumnExpression
from ..sql.elements import SQLCoreOperations
from ..util import FastIntFlag
from ..util.langhelpers import TypingOnly
from ..util.typing import Literal
def object_state(instance: _T) -> InstanceState[_T]:
    """Given an object, return the :class:`.InstanceState`
    associated with the object.

    Raises :class:`sqlalchemy.orm.exc.UnmappedInstanceError`
    if no mapping is configured.

    Equivalent functionality is available via the :func:`_sa.inspect`
    function as::

        inspect(instance)

    Using the inspection system will raise
    :class:`sqlalchemy.exc.NoInspectionAvailable` if the instance is
    not part of a mapping.

    """
    state = _inspect_mapped_object(instance)
    if state is None:
        raise exc.UnmappedInstanceError(instance)
    else:
        return state