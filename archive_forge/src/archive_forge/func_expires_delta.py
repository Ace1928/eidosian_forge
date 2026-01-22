from datetime import timedelta
from kombu.utils.objects import cached_property
from kombu.utils.url import _parse_url
from celery.exceptions import ImproperlyConfigured
from .base import KeyValueStoreBackend
@cached_property
def expires_delta(self):
    return timedelta(seconds=0 if self.expires is None else self.expires)