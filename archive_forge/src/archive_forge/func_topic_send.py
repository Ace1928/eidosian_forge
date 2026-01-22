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
def topic_send(self, exchange_name, topic, msg, timeout=None, retry=None, transport_options=None):
    """Send a 'topic' message."""
    exchange = kombu.entity.Exchange(name=exchange_name, type='topic', durable=self.durable, auto_delete=self.amqp_auto_delete)
    LOG.debug('Sending topic to %s with routing_key %s', exchange_name, topic)
    self._ensure_publishing(self._publish, exchange, msg, routing_key=topic, timeout=timeout, retry=retry, transport_options=transport_options)