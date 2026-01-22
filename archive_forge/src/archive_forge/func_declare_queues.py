import socket
from functools import partial
from itertools import cycle, islice
from kombu import Queue, eventloop
from kombu.common import maybe_declare
from kombu.utils.encoding import ensure_bytes
from celery.app import app_or_default
from celery.utils.nodenames import worker_direct
from celery.utils.text import str_to_list
def declare_queues(self, consumer):
    for queue in consumer.queues:
        if self.queues and queue.name not in self.queues:
            continue
        if self.on_declare_queue is not None:
            self.on_declare_queue(queue)
        try:
            _, mcount, _ = queue(consumer.channel).queue_declare(passive=True)
            if mcount:
                self.state.total_apx += mcount
        except self.conn.channel_errors:
            pass