import random
import ray
from typing import Any, Callable, Dict, Iterable, Optional, Set, Tuple, Union
def resolve_future(self, future: ray.ObjectRef):
    """Resolve a single future.

        This method will block until the future is available. It will then
        trigger the callback associated to the future and the event (success
        or error), if specified.

        Args:
            future: Ray future to resolve.

        """
    try:
        on_result, on_error = self._tracked_futures.pop(future)
    except KeyError as e:
        raise ValueError(f'Future {future} is not tracked by this RayEventManager') from e
    try:
        result = ray.get(future)
    except Exception as e:
        if on_error:
            on_error(e)
        else:
            raise e
    else:
        if on_result:
            on_result(result)