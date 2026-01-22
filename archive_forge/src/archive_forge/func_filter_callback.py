import socket
from functools import partial
from itertools import cycle, islice
from kombu import Queue, eventloop
from kombu.common import maybe_declare
from kombu.utils.encoding import ensure_bytes
from celery.app import app_or_default
from celery.utils.nodenames import worker_direct
from celery.utils.text import str_to_list
def filter_callback(callback, tasks):

    def filtered(body, message):
        if tasks and body['task'] not in tasks:
            return
        return callback(body, message)
    return filtered