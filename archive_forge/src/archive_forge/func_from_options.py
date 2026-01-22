import contextlib
import os
import re
import sys
import sentry_sdk
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk.utils import (
from sentry_sdk._compat import PY2, duration_in_milliseconds, iteritems
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.tracing import LOW_QUALITY_TRANSACTION_SOURCES
@classmethod
def from_options(cls, scope):
    sentry_items = {}
    third_party_items = ''
    mutable = False
    client = sentry_sdk.Hub.current.client
    if client is None or scope._propagation_context is None:
        return Baggage(sentry_items)
    options = client.options
    propagation_context = scope._propagation_context
    if propagation_context is not None and 'trace_id' in propagation_context:
        sentry_items['trace_id'] = propagation_context['trace_id']
    if options.get('environment'):
        sentry_items['environment'] = options['environment']
    if options.get('release'):
        sentry_items['release'] = options['release']
    if options.get('dsn'):
        sentry_items['public_key'] = Dsn(options['dsn']).public_key
    if options.get('traces_sample_rate'):
        sentry_items['sample_rate'] = options['traces_sample_rate']
    user = scope and scope._user or {}
    if user.get('segment'):
        sentry_items['user_segment'] = user['segment']
    return Baggage(sentry_items, third_party_items, mutable)