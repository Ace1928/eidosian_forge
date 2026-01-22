from collections import defaultdict, Counter
from typing import Dict, Generator, List, Optional, TypeVar
def has_cached_object(self, key: T) -> bool:
    """Return True if at least one cached object exists for this key.

        Args:
            key: Group key.

        Returns:
            True if at least one cached object exists for this key.
        """
    return bool(self._cached_objects[key])