from __future__ import absolute_import
import sys
from sentry_sdk._compat import reraise
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk import Hub
from sentry_sdk.consts import OP
from sentry_sdk.hub import _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.integrations.logging import ignore_logger
from sentry_sdk.tracing import Transaction, TRANSACTION_SOURCE_TASK
from sentry_sdk.utils import (
def patch_run_job():
    old_run_job = Worker.run_job

    async def _sentry_run_job(self, job_id, score):
        hub = Hub(Hub.current)
        if hub.get_integration(ArqIntegration) is None:
            return await old_run_job(self, job_id, score)
        with hub.push_scope() as scope:
            scope._name = 'arq'
            scope.clear_breadcrumbs()
            transaction = Transaction(name='unknown arq task', status='ok', op=OP.QUEUE_TASK_ARQ, source=TRANSACTION_SOURCE_TASK)
            with hub.start_transaction(transaction):
                return await old_run_job(self, job_id, score)
    Worker.run_job = _sentry_run_job