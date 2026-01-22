from __future__ import annotations
import os
import socket
import threading
from collections import deque
from contextlib import contextmanager
from functools import partial
from itertools import count
from uuid import NAMESPACE_OID, uuid3, uuid4, uuid5
from amqp import ChannelError, RecoverableConnectionError
from .entity import Exchange, Queue
from .log import get_logger
from .serialization import registry as serializers
from .utils.uuid import uuid
def send_reply(exchange, req, msg, producer=None, retry=False, retry_policy=None, **props):
    """Send reply for request.

    Arguments:
    ---------
        exchange (kombu.Exchange, str): Reply exchange
        req (~kombu.Message): Original request, a message with
            a ``reply_to`` property.
        producer (kombu.Producer): Producer instance
        retry (bool): If true must retry according to
            the ``reply_policy`` argument.
        retry_policy (Dict): Retry settings.
        **props (Any): Extra properties.
    """
    return producer.publish(msg, exchange=exchange, retry=retry, retry_policy=retry_policy, **dict({'routing_key': req.properties['reply_to'], 'correlation_id': req.properties.get('correlation_id'), 'serializer': serializers.type_to_name[req.content_type], 'content_encoding': req.content_encoding}, **props))