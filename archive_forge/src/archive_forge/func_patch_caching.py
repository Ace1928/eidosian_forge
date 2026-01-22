import functools
from typing import TYPE_CHECKING
from django import VERSION as DJANGO_VERSION
from django.core.cache import CacheHandler
from sentry_sdk import Hub
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk._compat import text_type
def patch_caching():
    from sentry_sdk.integrations.django import DjangoIntegration
    if not hasattr(CacheHandler, '_sentry_patched'):
        if DJANGO_VERSION < (3, 2):
            original_get_item = CacheHandler.__getitem__

            @functools.wraps(original_get_item)
            def sentry_get_item(self, alias):
                cache = original_get_item(self, alias)
                integration = Hub.current.get_integration(DjangoIntegration)
                if integration and integration.cache_spans:
                    _patch_cache(cache)
                return cache
            CacheHandler.__getitem__ = sentry_get_item
            CacheHandler._sentry_patched = True
        else:
            original_create_connection = CacheHandler.create_connection

            @functools.wraps(original_create_connection)
            def sentry_create_connection(self, alias):
                cache = original_create_connection(self, alias)
                integration = Hub.current.get_integration(DjangoIntegration)
                if integration and integration.cache_spans:
                    _patch_cache(cache)
                return cache
            CacheHandler.create_connection = sentry_create_connection
            CacheHandler._sentry_patched = True