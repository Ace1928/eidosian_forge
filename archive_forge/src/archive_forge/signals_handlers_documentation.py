from __future__ import absolute_import
from django.dispatch import Signal
from sentry_sdk import Hub
from sentry_sdk._functools import wraps
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import OP
from sentry_sdk.integrations.django import DJANGO_VERSION

    Patch django signal receivers to create a span.

    This only wraps sync receivers. Django>=5.0 introduced async receivers, but
    since we don't create transactions for ASGI Django, we don't wrap them.
    