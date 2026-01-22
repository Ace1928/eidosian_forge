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
def marshal_request(request, context, envelope=False, call_monitor_timeout=None):
    msg = proton.Message(inferred=True)
    if envelope:
        request = common.serialize_msg(request)
    data = {'request': request, 'context': context}
    if call_monitor_timeout is not None:
        data['call_monitor_timeout'] = call_monitor_timeout
    msg.body = jsonutils.dumps(data)
    return msg