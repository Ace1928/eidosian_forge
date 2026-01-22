import collections
import contextlib
import errno
import functools
import itertools
import math
import os
import random
import socket
import ssl
import sys
import threading
import time
from urllib import parse
import uuid
from amqp import exceptions as amqp_ex
import kombu
import kombu.connection
import kombu.entity
import kombu.messaging
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import eventletutils
import oslo_messaging
from oslo_messaging._drivers import amqp as rpc_amqp
from oslo_messaging._drivers import amqpdriver
from oslo_messaging._drivers import base
from oslo_messaging._drivers import common as rpc_common
from oslo_messaging._drivers import pool
from oslo_messaging import _utils
from oslo_messaging import exceptions
def declare_fanout_consumer(self, topic, callback):
    """Create a 'fanout' consumer."""
    exchange_name = '%s_fanout' % topic
    if self.rabbit_stream_fanout:
        queue_name = '%s_fanout' % topic
    else:
        if self._q_manager:
            unique = self._q_manager.get()
        else:
            unique = uuid.uuid4().hex
        queue_name = '%s_fanout_%s' % (topic, unique)
    LOG.info('Creating fanout queue: %s', queue_name)
    is_durable = self.rabbit_transient_quorum_queue or self.rabbit_stream_fanout
    consumer = Consumer(exchange_name=exchange_name, queue_name=queue_name, routing_key=topic, type='fanout', durable=is_durable, exchange_auto_delete=True, queue_auto_delete=False, callback=callback, rabbit_ha_queues=self.rabbit_ha_queues, rabbit_queue_ttl=self.rabbit_transient_queues_ttl, enable_cancel_on_failover=self.enable_cancel_on_failover, rabbit_quorum_queue=self.rabbit_transient_quorum_queue, rabbit_quorum_queue_config=self.rabbit_quorum_queue_config, rabbit_stream_fanout=self.rabbit_stream_fanout)
    self.declare_consumer(consumer)