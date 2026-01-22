from django.core import signals
from django.db.utils import (
from django.utils.connection import ConnectionProxy
def reset_queries(**kwargs):
    for conn in connections.all(initialized_only=True):
        conn.queries_log.clear()