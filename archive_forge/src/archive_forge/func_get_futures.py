import random
import ray
from typing import Any, Callable, Dict, Iterable, Optional, Set, Tuple, Union
def get_futures(self) -> Set[ray.ObjectRef]:
    """Get futures tracked by the event manager."""
    return set(self._tracked_futures)