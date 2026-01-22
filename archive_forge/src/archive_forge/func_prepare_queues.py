import socket
from functools import partial
from itertools import cycle, islice
from kombu import Queue, eventloop
from kombu.common import maybe_declare
from kombu.utils.encoding import ensure_bytes
from celery.app import app_or_default
from celery.utils.nodenames import worker_direct
from celery.utils.text import str_to_list
def prepare_queues(queues):
    if isinstance(queues, str):
        queues = queues.split(',')
    if isinstance(queues, list):
        queues = dict((tuple(islice(cycle(q.split(':')), None, 2)) for q in queues))
    if queues is None:
        queues = {}
    return queues