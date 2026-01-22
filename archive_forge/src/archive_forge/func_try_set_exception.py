import threading
from concurrent.futures import Future
from typing import Any, Callable, Generator, Generic, Optional, Tuple, Type, TypeVar
def try_set_exception(self, exception: Optional[BaseException]) -> bool:
    """Sets an exception on this future if not already done.

        Returns:
            True if we set the exception, False if the future was already done.
        """
    with self._condition:
        if self.done():
            return False
        self.set_exception(exception)
        return True