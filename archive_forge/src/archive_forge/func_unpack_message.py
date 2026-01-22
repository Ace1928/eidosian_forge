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
def unpack_message(msg):
    """Unpack context and msg."""
    context = {}
    message = None
    msg = jsonutils.loads(msg)
    message = driver_common.deserialize_msg(msg)
    context = message['_context']
    del message['_context']
    return (context, message)