import contextlib
from typing import Callable, Generator, Type
@contextlib.contextmanager
def convertError(sourceType: Type[BaseException], targetType: Callable[[], BaseException]) -> Generator[None, None, None]:
    """
    Convert an error into a different error type.

    @param sourceType: The type of exception that should be caught and
        converted.
    @type sourceType: L{BaseException}

    @param targetType: The type of exception to which the original should be
        converted.
    @type targetType: L{BaseException}
    """
    try:
        yield
    except sourceType as e:
        raise targetType().with_traceback(e.__traceback__)