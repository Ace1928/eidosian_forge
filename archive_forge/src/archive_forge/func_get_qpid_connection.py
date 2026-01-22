from __future__ import annotations
import os
import select
import socket
import ssl
import sys
import uuid
from gettext import gettext as _
from queue import Empty
from time import monotonic
import amqp.protocol
from kombu.log import get_logger
from kombu.transport import base, virtual
from kombu.transport.virtual import Base64, Message
def get_qpid_connection(self):
    """Return the existing connection (singleton).

        :return: The existing qpid.messaging.Connection
        :rtype: :class:`qpid.messaging.endpoints.Connection`

        """
    return self._qpid_conn