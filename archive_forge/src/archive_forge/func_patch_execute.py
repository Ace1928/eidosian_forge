from __future__ import absolute_import
import sys
from datetime import datetime
from sentry_sdk._compat import reraise
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk import Hub
from sentry_sdk.api import continue_trace, get_baggage, get_traceparent
from sentry_sdk.consts import OP
from sentry_sdk.hub import _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.tracing import (
from sentry_sdk.utils import (
def patch_execute():
    old_execute = Huey._execute

    def _sentry_execute(self, task, timestamp=None):
        hub = Hub.current
        if hub.get_integration(HueyIntegration) is None:
            return old_execute(self, task, timestamp)
        with hub.push_scope() as scope:
            with capture_internal_exceptions():
                scope._name = 'huey'
                scope.clear_breadcrumbs()
                scope.add_event_processor(_make_event_processor(task))
            sentry_headers = task.kwargs.pop('sentry_headers', None)
            transaction = continue_trace(sentry_headers or {}, name=task.name, op=OP.QUEUE_TASK_HUEY, source=TRANSACTION_SOURCE_TASK)
            transaction.set_status('ok')
            if not getattr(task, '_sentry_is_patched', False):
                task.execute = _wrap_task_execute(task.execute)
                task._sentry_is_patched = True
            with hub.start_transaction(transaction):
                return old_execute(self, task, timestamp)
    Huey._execute = _sentry_execute