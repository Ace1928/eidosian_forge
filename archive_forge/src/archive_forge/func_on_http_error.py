import functools
from typing import Any, Callable, cast, Optional, Type, TYPE_CHECKING, TypeVar, Union
def on_http_error(error: Type[Exception]) -> Callable[[__F], __F]:
    """Manage GitlabHttpError exceptions.

    This decorator function can be used to catch GitlabHttpError exceptions
    raise specialized exceptions instead.

    Args:
        The exception type to raise -- must inherit from GitlabError
    """

    def wrap(f: __F) -> __F:

        @functools.wraps(f)
        def wrapped_f(*args: Any, **kwargs: Any) -> Any:
            try:
                return f(*args, **kwargs)
            except GitlabHttpError as e:
                raise error(e.error_message, e.response_code, e.response_body) from e
        return cast(__F, wrapped_f)
    return wrap