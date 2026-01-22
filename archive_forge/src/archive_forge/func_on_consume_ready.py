import time
from operator import itemgetter
from kombu import Queue
from kombu.connection import maybe_channel
from kombu.mixins import ConsumerMixin
from celery import uuid
from celery.app import app_or_default
from celery.utils.time import adjust_timestamp
from .event import get_exchange
def on_consume_ready(self, connection, channel, consumers, wakeup=True, **kwargs):
    if wakeup:
        self.wakeup_workers(channel=channel)