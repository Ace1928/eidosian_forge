from collections import defaultdict
from functools import partial
from heapq import heappush
from operator import itemgetter
from kombu import Consumer
from kombu.asynchronous.semaphore import DummyLock
from kombu.exceptions import ContentDisallowed, DecodeError
from celery import bootsteps
from celery.utils.log import get_logger
from celery.utils.objects import Bunch
from .mingle import Mingle
def on_elect(self, event):
    try:
        id_, clock, hostname, pid, topic, action, _ = self._cons_stamp_fields(event)
    except KeyError as exc:
        return logger.exception('election request missing field %s', exc)
    heappush(self.consensus_requests[id_], (clock, f'{hostname}.{pid}', topic, action))
    self.dispatcher.send('worker-elect-ack', id=id_)