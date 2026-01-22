from __future__ import annotations
from types import TracebackType
from typing import Callable
from typing import cast
from typing import final
from typing import Generic
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TypeVar
def force_result(self, result: ResultType) -> None:
    """Force the result(s) to ``result``.

        If the hook was marked as a ``firstresult`` a single value should
        be set, otherwise set a (modified) list of results. Any exceptions
        found during invocation will be deleted.

        This overrides any previous result or exception.
        """
    self._result = result
    self._exception = None