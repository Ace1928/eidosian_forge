import threading
from concurrent.futures import Future
from typing import Any, Callable, Generator, Generic, Optional, Tuple, Type, TypeVar
def failed_future(error: BaseException) -> AwaitableFuture[Any]:
    """Return a future that will fail with the given error."""
    f = AwaitableFuture[Any]()
    f.set_exception(error)
    return f