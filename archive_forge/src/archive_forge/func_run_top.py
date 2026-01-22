import itertools
from functools import update_wrapper
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from .entry_points import load_entry_point
def run_top(self, *args: Any, **kwargs: Any) -> Any:
    """Execute the first matching child function

        :return: the return of the child function
        """
    return list(itertools.islice(self.run(*args, **kwargs), 1))[0]