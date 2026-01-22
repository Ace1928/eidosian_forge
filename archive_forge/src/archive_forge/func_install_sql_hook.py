from __future__ import absolute_import
import inspect
import sys
import threading
import weakref
from importlib import import_module
from sentry_sdk._compat import string_types, text_type
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk.db.explain_plan.django import attach_explain_plan_to_span
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.scope import add_global_event_processor
from sentry_sdk.serializer import add_global_repr_processor
from sentry_sdk.tracing import SOURCE_FOR_STYLE, TRANSACTION_SOURCE_URL
from sentry_sdk.tracing_utils import add_query_source, record_sql_queries
from sentry_sdk.utils import (
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.integrations.logging import ignore_logger
from sentry_sdk.integrations.wsgi import SentryWsgiMiddleware
from sentry_sdk.integrations._wsgi_common import RequestExtractor
from sentry_sdk.integrations.django.transactions import LEGACY_RESOLVER
from sentry_sdk.integrations.django.templates import (
from sentry_sdk.integrations.django.middleware import patch_django_middlewares
from sentry_sdk.integrations.django.signals_handlers import patch_signals
from sentry_sdk.integrations.django.views import patch_views
def install_sql_hook():
    """If installed this causes Django's queries to be captured."""
    try:
        from django.db.backends.utils import CursorWrapper
    except ImportError:
        from django.db.backends.util import CursorWrapper
    try:
        from django.db.backends import BaseDatabaseWrapper
    except ImportError:
        from django.db.backends.base.base import BaseDatabaseWrapper
    try:
        real_execute = CursorWrapper.execute
        real_executemany = CursorWrapper.executemany
        real_connect = BaseDatabaseWrapper.connect
    except AttributeError:
        return

    def execute(self, sql, params=None):
        hub = Hub.current
        if hub.get_integration(DjangoIntegration) is None:
            return real_execute(self, sql, params)
        with record_sql_queries(hub, self.cursor, sql, params, paramstyle='format', executemany=False) as span:
            _set_db_data(span, self)
            if hub.client:
                options = hub.client.options['_experiments'].get('attach_explain_plans')
                if options is not None:
                    attach_explain_plan_to_span(span, self.cursor.connection, sql, params, self.mogrify, options)
            result = real_execute(self, sql, params)
        with capture_internal_exceptions():
            add_query_source(hub, span)
        return result

    def executemany(self, sql, param_list):
        hub = Hub.current
        if hub.get_integration(DjangoIntegration) is None:
            return real_executemany(self, sql, param_list)
        with record_sql_queries(hub, self.cursor, sql, param_list, paramstyle='format', executemany=True) as span:
            _set_db_data(span, self)
            result = real_executemany(self, sql, param_list)
        with capture_internal_exceptions():
            add_query_source(hub, span)
        return result

    def connect(self):
        hub = Hub.current
        if hub.get_integration(DjangoIntegration) is None:
            return real_connect(self)
        with capture_internal_exceptions():
            hub.add_breadcrumb(message='connect', category='query')
        with hub.start_span(op=OP.DB, description='connect') as span:
            _set_db_data(span, self)
            return real_connect(self)
    CursorWrapper.execute = execute
    CursorWrapper.executemany = executemany
    BaseDatabaseWrapper.connect = connect
    ignore_logger('django.db.backends')