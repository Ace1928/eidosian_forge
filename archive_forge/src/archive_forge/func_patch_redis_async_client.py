from __future__ import absolute_import
from sentry_sdk import Hub
from sentry_sdk.consts import OP
from sentry_sdk.integrations.redis import (
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.tracing import Span
from sentry_sdk.utils import capture_internal_exceptions
def patch_redis_async_client(cls, is_cluster, set_db_data_fn):
    old_execute_command = cls.execute_command

    async def _sentry_execute_command(self, name, *args, **kwargs):
        hub = Hub.current
        if hub.get_integration(RedisIntegration) is None:
            return await old_execute_command(self, name, *args, **kwargs)
        description = _get_span_description(name, *args)
        with hub.start_span(op=OP.DB_REDIS, description=description) as span:
            set_db_data_fn(span, self)
            _set_client_data(span, is_cluster, name, *args)
            return await old_execute_command(self, name, *args, **kwargs)
    cls.execute_command = _sentry_execute_command