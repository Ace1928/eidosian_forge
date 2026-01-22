import collections
import logging
import os
import threading
import uuid
import warnings
from debtcollector import removals
from oslo_config import cfg
from oslo_messaging.target import Target
from oslo_serialization import jsonutils
from oslo_utils import importutils
from oslo_utils import timeutils
from oslo_messaging._drivers.amqp1_driver.eventloop import compute_timeout
from oslo_messaging._drivers.amqp1_driver import opts
from oslo_messaging._drivers import base
from oslo_messaging._drivers import common
def unmarshal_request(message):
    data = jsonutils.loads(message.body)
    msg = common.deserialize_msg(data.get('request'))
    return (msg, data.get('context'), data.get('call_monitor_timeout'))