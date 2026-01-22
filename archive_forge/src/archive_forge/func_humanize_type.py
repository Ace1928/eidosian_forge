import sys
from datetime import datetime
from celery.app import app_or_default
from celery.utils.functional import LRUCache
from celery.utils.time import humanize_seconds
def humanize_type(type):
    try:
        return HUMAN_TYPES[type.lower()]
    except KeyError:
        return type.lower().replace('-', ' ')