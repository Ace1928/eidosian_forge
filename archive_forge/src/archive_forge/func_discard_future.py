import random
import ray
from typing import Any, Callable, Dict, Iterable, Optional, Set, Tuple, Union
def discard_future(self, future: ray.ObjectRef):
    """Remove future from tracking.

        The future will not be awaited anymore, and it will not trigger any callbacks.

        Args:
            future: Ray futures to discard.
        """
    self._tracked_futures.pop(future, None)