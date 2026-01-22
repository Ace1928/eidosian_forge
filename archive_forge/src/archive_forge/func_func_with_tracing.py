import inspect
from functools import wraps
import sentry_sdk
from sentry_sdk import get_current_span
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import OP
from sentry_sdk.utils import logger, qualname_from_function
@wraps(func)
def func_with_tracing(*args, **kwargs):
    span = get_current_span(sentry_sdk.Hub.current)
    if span is None:
        logger.warning('Can not create a child span for %s. Please start a Sentry transaction before calling this function.', qualname_from_function(func))
        return func(*args, **kwargs)
    with span.start_child(op=OP.FUNCTION, description=qualname_from_function(func)):
        return func(*args, **kwargs)