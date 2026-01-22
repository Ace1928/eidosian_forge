from collections import defaultdict, Counter
from typing import Dict, Generator, List, Optional, TypeVar
def decrease_max(self, key: T, by: int=1) -> None:
    """Decrease number of max objects for this key.

        Args:
            key: Group key.
            by: Decrease by this amount.
        """
    self._max_num_objects[key] -= by