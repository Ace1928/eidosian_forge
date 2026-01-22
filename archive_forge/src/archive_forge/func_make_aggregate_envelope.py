import os
import time
from threading import Thread, Lock
from contextlib import contextmanager
import sentry_sdk
from sentry_sdk.envelope import Envelope
from sentry_sdk.session import Session
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import format_timestamp
def make_aggregate_envelope(aggregate_states, attrs):
    return {'attrs': dict(attrs), 'aggregates': list(aggregate_states.values())}