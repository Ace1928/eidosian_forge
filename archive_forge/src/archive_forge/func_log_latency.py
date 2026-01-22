import contextlib
import logging
import threading
import time
from tensorboard.util import tb_logging
def log_latency(region_name_or_function_to_decorate, log_level=None):
    """Log latency in a function or region.

    Three usages are supported. As a decorator:

    >>> @log_latency
    ... def function_1():
    ...     pass
    ...


    As a decorator with a custom label for the region:

    >>> @log_latency("custom_label")
    ... def function_2():
    ...     pass
    ...

    As a context manager:

    >>> def function_3():
    ...     with log_latency("region_within_function"):
    ...         pass
    ...

    Args:
        region_name_or_function_to_decorate: Either: a `str`, in which
            case the result of this function may be used as either a
            decorator or a context manager; or a callable, in which case
            the result of this function is a decorated version of that
            callable.
        log_level: Optional integer logging level constant. Defaults to
            `logging.INFO`.

    Returns:
        A decorated version of the input callable, or a dual
        decorator/context manager with the input region name.
    """
    if log_level is None:
        log_level = logging.INFO
    if isinstance(region_name_or_function_to_decorate, str):
        region_name = region_name_or_function_to_decorate
        return _log_latency(region_name, log_level)
    else:
        function_to_decorate = region_name_or_function_to_decorate
        qualname = getattr(function_to_decorate, '__qualname__', None)
        if qualname is None:
            qualname = str(function_to_decorate)
        decorator = _log_latency(qualname, log_level)
        return decorator(function_to_decorate)