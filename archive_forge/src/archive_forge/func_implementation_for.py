from __future__ import annotations
from contextlib import contextmanager
import re
import textwrap
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import Dict
from typing import Iterator
from typing import List  # noqa
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence  # noqa
from typing import Tuple
from typing import Type  # noqa
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from sqlalchemy.sql.elements import conv
from . import batch
from . import schemaobj
from .. import util
from ..util import sqla_compat
from ..util.compat import formatannotation_fwdref
from ..util.compat import inspect_formatargspec
from ..util.compat import inspect_getfullargspec
from ..util.sqla_compat import _literal_bindparam
@classmethod
def implementation_for(cls, op_cls: Any) -> Callable[[_C], _C]:
    """Register an implementation for a given :class:`.MigrateOperation`.

        This is part of the operation extensibility API.

        .. seealso::

            :ref:`operation_plugins` - example of use

        """

    def decorate(fn: _C) -> _C:
        cls._to_impl.dispatch_for(op_cls)(fn)
        return fn
    return decorate