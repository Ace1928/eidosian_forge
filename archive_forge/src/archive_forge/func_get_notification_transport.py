import abc
import argparse
import logging
import uuid
from oslo_config import cfg
from oslo_utils import timeutils
from stevedore import extension
from stevedore import named
from oslo_messaging import serializer as msg_serializer
from oslo_messaging import transport as msg_transport
def get_notification_transport(conf, url=None, allowed_remote_exmods=None):
    """A factory method for Transport objects for notifications.

    This method should be used for notifications, in case notifications are
    being sent over a different message bus than normal messaging
    functionality; for example, using a different driver, or with different
    access permissions.

    If no transport URL is provided, the URL in the notifications section of
    the config file will be used.  If that URL is also absent, the same
    transport as specified in the messaging section will be used.

    If a transport URL is provided, then this function works exactly the same
    as get_transport.

    :param conf: the user configuration
    :type conf: cfg.ConfigOpts
    :param url: a transport URL, see :py:class:`transport.TransportURL`
    :type url: str or TransportURL
    :param allowed_remote_exmods: a list of modules which a client using this
                                  transport will deserialize remote exceptions
                                  from
    :type allowed_remote_exmods: list
    """
    conf.register_opts(_notifier_opts, group='oslo_messaging_notifications')
    if url is None:
        url = conf.oslo_messaging_notifications.transport_url
    return msg_transport._get_transport(conf, url, allowed_remote_exmods, transport_cls=msg_transport.NotificationTransport)