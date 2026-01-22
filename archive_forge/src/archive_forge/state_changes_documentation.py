from __future__ import annotations
import contextlib
from enum import Enum
from typing import Any
from typing import Callable
from typing import cast
from typing import Iterator
from typing import NoReturn
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union
from .. import exc as sa_exc
from .. import util
from ..util.typing import Literal
called within a method that changes states.

        method must also use the ``@declare_states()`` decorator.

        