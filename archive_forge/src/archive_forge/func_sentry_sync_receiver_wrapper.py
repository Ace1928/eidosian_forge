from __future__ import absolute_import
from django.dispatch import Signal
from sentry_sdk import Hub
from sentry_sdk._functools import wraps
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import OP
from sentry_sdk.integrations.django import DJANGO_VERSION
def sentry_sync_receiver_wrapper(receiver):

    @wraps(receiver)
    def wrapper(*args, **kwargs):
        signal_name = _get_receiver_name(receiver)
        with hub.start_span(op=OP.EVENT_DJANGO, description=signal_name) as span:
            span.set_data('signal', signal_name)
            return receiver(*args, **kwargs)
    return wrapper