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
def patch_enqueue_job():
    old_enqueue_job = ArqRedis.enqueue_job

    async def _sentry_enqueue_job(self, function, *args, **kwargs):
        hub = Hub.current
        if hub.get_integration(ArqIntegration) is None:
            return await old_enqueue_job(self, function, *args, **kwargs)
        with hub.start_span(op=OP.QUEUE_SUBMIT_ARQ, description=function):
            return await old_enqueue_job(self, function, *args, **kwargs)
    ArqRedis.enqueue_job = _sentry_enqueue_job