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
def listen_for_notifications(self, targets_and_priorities, pool, batch_size, batch_timeout):
    """Listen to a specified list of targets on Kafka brokers

        :param targets_and_priorities: List of pairs (target, priority)
                                       priority is not used for kafka driver
                                       target.exchange_target.topic is used as
                                       a kafka topic
        :type targets_and_priorities: list
        :param pool: consumer group of Kafka consumers
        :type pool: string
        """
    conn = ConsumerConnection(self.conf, self._url)
    topics = []
    for target, priority in targets_and_priorities:
        topics.append(target_to_topic(target, priority))
    conn.declare_topic_consumer(topics, pool)
    listener = KafkaListener(conn)
    return base.PollStyleListenerAdapter(listener, batch_size, batch_timeout)