from __future__ import annotations
import collections
import types
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Deque
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union
import weakref
from .. import exc
from .. import util
def with_wrapper(self, fn_wrap: _ListenerFnType) -> _EventKey[_ET]:
    if fn_wrap is self._listen_fn:
        return self
    else:
        return _EventKey(self.target, self.identifier, self.fn, self.dispatch_target, _fn_wrap=fn_wrap)