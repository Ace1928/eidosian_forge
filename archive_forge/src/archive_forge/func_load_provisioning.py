from __future__ import annotations
from enum import Enum
from types import ModuleType
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import ClassVar
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .. import util
from ..event import EventTarget
from ..pool import Pool
from ..pool import PoolProxiedConnection
from ..sql.compiler import Compiled as Compiled
from ..sql.compiler import Compiled  # noqa
from ..sql.compiler import TypeCompiler as TypeCompiler
from ..sql.compiler import TypeCompiler  # noqa
from ..util import immutabledict
from ..util.concurrency import await_only
from ..util.typing import Literal
from ..util.typing import NotRequired
from ..util.typing import Protocol
from ..util.typing import TypedDict
@classmethod
def load_provisioning(cls) -> None:
    """set up the provision.py module for this dialect.

        For dialects that include a provision.py module that sets up
        provisioning followers, this method should initiate that process.

        A typical implementation would be::

            @classmethod
            def load_provisioning(cls):
                __import__("mydialect.provision")

        The default method assumes a module named ``provision.py`` inside
        the owning package of the current dialect, based on the ``__module__``
        attribute::

            @classmethod
            def load_provisioning(cls):
                package = ".".join(cls.__module__.split(".")[0:-1])
                try:
                    __import__(package + ".provision")
                except ImportError:
                    pass

        .. versionadded:: 1.3.14

        """