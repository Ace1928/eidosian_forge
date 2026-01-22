import logging
import threading
import confluent_kafka
from confluent_kafka import KafkaException
from oslo_serialization import jsonutils
from oslo_utils import eventletutils
from oslo_utils import importutils
from oslo_messaging._drivers import base
from oslo_messaging._drivers import common as driver_common
from oslo_messaging._drivers.kafka_driver import kafka_options
def notify_send(self, topic, ctxt, msg, retry):
    """Send messages to Kafka broker.

        :param topic: String of the topic
        :param ctxt: context for the messages
        :param msg: messages for publishing
        :param retry: the number of retry
        """
    retry = retry if retry >= 0 else None
    message = pack_message(ctxt, msg)
    message = jsonutils.dumps(message).encode('utf-8')
    try:
        self._ensure_producer()
        poll = 0
        while True:
            try:
                if eventletutils.is_monkey_patched('thread'):
                    return tpool.execute(self._produce_message, topic, message, poll)
                return self._produce_message(topic, message, poll)
            except KafkaException as e:
                LOG.error('Produce message failed: %s' % str(e))
                break
            except BufferError:
                LOG.debug('Produce message queue full, waiting for deliveries')
                poll = 0.5
    except Exception:
        self._close_producer()
        raise