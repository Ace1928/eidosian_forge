import socket
from functools import partial
from itertools import cycle, islice
from kombu import Queue, eventloop
from kombu.common import maybe_declare
from kombu.utils.encoding import ensure_bytes
from celery.app import app_or_default
from celery.utils.nodenames import worker_direct
from celery.utils.text import str_to_list
def prepare_consumer(self, consumer):
    filter = self.filter
    update_state = self.update_state
    ack_message = self.ack_message
    if self.tasks:
        filter = filter_callback(filter, self.tasks)
        update_state = filter_callback(update_state, self.tasks)
        ack_message = filter_callback(ack_message, self.tasks)
    consumer.register_callback(filter)
    consumer.register_callback(update_state)
    if self.ack_messages:
        consumer.register_callback(self.ack_message)
    if self.callback is not None:
        callback = partial(self.callback, self.state)
        if self.tasks:
            callback = filter_callback(callback, self.tasks)
        consumer.register_callback(callback)
    self.declare_queues(consumer)
    return consumer