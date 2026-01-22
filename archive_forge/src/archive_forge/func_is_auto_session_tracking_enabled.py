import os
import time
from threading import Thread, Lock
from contextlib import contextmanager
import sentry_sdk
from sentry_sdk.envelope import Envelope
from sentry_sdk.session import Session
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import format_timestamp
def is_auto_session_tracking_enabled(hub=None):
    """Utility function to find out if session tracking is enabled."""
    if hub is None:
        hub = sentry_sdk.Hub.current
    should_track = hub.scope._force_auto_session_tracking
    if should_track is None:
        client_options = hub.client.options if hub.client else {}
        should_track = client_options.get('auto_session_tracking', False)
    return should_track