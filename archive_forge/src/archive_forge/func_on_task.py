import socket
from functools import partial
from itertools import cycle, islice
from kombu import Queue, eventloop
from kombu.common import maybe_declare
from kombu.utils.encoding import ensure_bytes
from celery.app import app_or_default
from celery.utils.nodenames import worker_direct
from celery.utils.text import str_to_list
def on_task(body, message):
    ret = predicate(body, message)
    if ret:
        if transform:
            ret = transform(ret)
        if isinstance(ret, Queue):
            maybe_declare(ret, conn.default_channel)
            ex, rk = (ret.exchange.name, ret.routing_key)
        else:
            ex, rk = expand_dest(ret, exchange, routing_key)
        republish(producer, message, exchange=ex, routing_key=rk)
        message.ack()
        state.filtered += 1
        if callback:
            callback(state, body, message)
        if limit and state.filtered >= limit:
            raise StopFiltering()