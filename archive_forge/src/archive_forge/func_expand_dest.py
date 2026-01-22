import socket
from functools import partial
from itertools import cycle, islice
from kombu import Queue, eventloop
from kombu.common import maybe_declare
from kombu.utils.encoding import ensure_bytes
from celery.app import app_or_default
from celery.utils.nodenames import worker_direct
from celery.utils.text import str_to_list
def expand_dest(ret, exchange, routing_key):
    try:
        ex, rk = ret
    except (TypeError, ValueError):
        ex, rk = (exchange, routing_key)
    return (ex, rk)