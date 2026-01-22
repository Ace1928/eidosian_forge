import random
import ray
from typing import Any, Callable, Dict, Iterable, Optional, Set, Tuple, Union
def track_futures(self, futures: Iterable[ray.ObjectRef], on_result: Optional[_ResultCallback]=None, on_error: Optional[_ErrorCallback]=None):
    """Track multiple futures and invoke callbacks on resolution.

        Control has to be yielded to the event manager for the callbacks to
        be invoked, either via :meth:`wait` or via :meth:`resolve_future`.

        Args:
            futures: Ray futures to await.
            on_result: Callback to invoke when the future resolves successfully.
            on_error: Callback to invoke when the future fails.

        """
    for future in futures:
        self.track_future(future, on_result=on_result, on_error=on_error)