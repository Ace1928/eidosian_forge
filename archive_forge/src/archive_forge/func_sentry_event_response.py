import sys
from sentry_sdk._compat import reraise
from sentry_sdk.hub import Hub
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.integrations.aws_lambda import _make_request_event_processor
from sentry_sdk.tracing import TRANSACTION_SOURCE_COMPONENT
from sentry_sdk.utils import (
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk._functools import wraps
import chalice  # type: ignore
from chalice import Chalice, ChaliceViewError
from chalice.app import EventSourceHandler as ChaliceEventSourceHandler  # type: ignore
def sentry_event_response(app, view_function, function_args):
    wrapped_view_function = _get_view_function_response(app, view_function, function_args)
    return old_get_view_function_response(app, wrapped_view_function, function_args)