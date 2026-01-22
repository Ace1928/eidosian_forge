from django.core import signals
from django.core.cache.backends.base import (
from django.utils.connection import BaseConnectionHandler, ConnectionProxy
from django.utils.module_loading import import_string
def close_caches(**kwargs):
    caches.close_all()