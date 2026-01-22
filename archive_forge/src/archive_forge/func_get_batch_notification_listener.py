import itertools
import logging
from oslo_messaging.notify import dispatcher as notify_dispatcher
from oslo_messaging import server as msg_server
from oslo_messaging import transport as msg_transport
def get_batch_notification_listener(transport, targets, endpoints, executor=None, serializer=None, allow_requeue=False, pool=None, batch_size=None, batch_timeout=None):
    """Construct a batch notification listener

    The executor parameter controls how incoming messages will be received and
    dispatched.

    If the eventlet executor is used, the threading and time library need to be
    monkeypatched.

    :param transport: the messaging transport
    :type transport: Transport
    :param targets: the exchanges and topics to listen on
    :type targets: list of Target
    :param endpoints: a list of endpoint objects
    :type endpoints: list
    :param executor: name of message executor - available values are
                     'eventlet' and 'threading'
    :type executor: str
    :param serializer: an optional entity serializer
    :type serializer: Serializer
    :param allow_requeue: whether NotificationResult.REQUEUE support is needed
    :type allow_requeue: bool
    :param pool: the pool name
    :type pool: str
    :param batch_size: number of messages to wait before calling
                       endpoints callacks
    :type batch_size: int
    :param batch_timeout: number of seconds to wait before calling
                       endpoints callacks
    :type batch_timeout: int
    :raises: NotImplementedError
    """
    dispatcher = notify_dispatcher.BatchNotificationDispatcher(endpoints, serializer)
    return BatchNotificationServer(transport, targets, dispatcher, executor, allow_requeue, pool, batch_size, batch_timeout)