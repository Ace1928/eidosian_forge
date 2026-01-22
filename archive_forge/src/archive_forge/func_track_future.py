import random
import ray
from typing import Any, Callable, Dict, Iterable, Optional, Set, Tuple, Union
def track_future(self, future: ray.ObjectRef, on_result: Optional[_ResultCallback]=None, on_error: Optional[_ErrorCallback]=None):
    """Track a single future and invoke callbacks on resolution.

        Control has to be yielded to the event manager for the callbacks to
        be invoked, either via :meth:`wait` or via :meth:`resolve_future`.

        Args:
            future: Ray future to await.
            on_result: Callback to invoke when the future resolves successfully.
            on_error: Callback to invoke when the future fails.

        """
    self._tracked_futures[future] = (on_result, on_error)