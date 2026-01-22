import socket
from functools import partial
from itertools import cycle, islice
from kombu import Queue, eventloop
from kombu.common import maybe_declare
from kombu.utils.encoding import ensure_bytes
from celery.app import app_or_default
from celery.utils.nodenames import worker_direct
from celery.utils.text import str_to_list
def migrate_task(producer, body_, message, queues=None):
    """Migrate single task message."""
    info = message.delivery_info
    queues = {} if queues is None else queues
    republish(producer, message, exchange=queues.get(info['exchange']), routing_key=queues.get(info['routing_key']))