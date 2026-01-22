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
def get_consumers(self, channel):
    self.register_timer()
    ev = self.Receiver(channel, routing_key='worker.#', queue_ttl=self.heartbeat_interval)
    return [Consumer(channel, queues=[ev.queue], on_message=partial(self.on_message, ev.event_from_message), accept=ev.accept, no_ack=True)]