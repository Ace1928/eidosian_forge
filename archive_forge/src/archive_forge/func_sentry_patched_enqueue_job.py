from __future__ import absolute_import
import weakref
from sentry_sdk.consts import OP
from sentry_sdk.api import continue_trace
from sentry_sdk.hub import Hub
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.integrations.logging import ignore_logger
from sentry_sdk.tracing import TRANSACTION_SOURCE_TASK
from sentry_sdk.utils import (
from sentry_sdk._types import TYPE_CHECKING
def sentry_patched_enqueue_job(self, job, **kwargs):
    hub = Hub.current
    if hub.get_integration(RqIntegration) is not None:
        if hub.scope.span is not None:
            job.meta['_sentry_trace_headers'] = dict(hub.iter_trace_propagation_headers())
    return old_enqueue_job(self, job, **kwargs)