import inspect
from functools import wraps
import sentry_sdk
from sentry_sdk import get_current_span
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import OP
from sentry_sdk.utils import logger, qualname_from_function

    Decorator to add child spans for functions.

    This is the Python 3 compatible version of the decorator.
    For Python 2 there is duplicated code here: ``sentry_sdk.tracing_utils_python2.start_child_span_decorator()``.

    See also ``sentry_sdk.tracing.trace()``.
    