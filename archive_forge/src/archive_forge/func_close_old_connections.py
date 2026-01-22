from django.core import signals
from django.db.utils import (
from django.utils.connection import ConnectionProxy
def close_old_connections(**kwargs):
    for conn in connections.all(initialized_only=True):
        conn.close_if_unusable_or_obsolete()