import time
from operator import itemgetter
from kombu import Queue
from kombu.connection import maybe_channel
from kombu.mixins import ConsumerMixin
from celery import uuid
from celery.app import app_or_default
from celery.utils.time import adjust_timestamp
from .event import get_exchange
def itercapture(self, limit=None, timeout=None, wakeup=True):
    return self.consume(limit=limit, timeout=timeout, wakeup=wakeup)