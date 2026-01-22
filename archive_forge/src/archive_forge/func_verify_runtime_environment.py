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
def verify_runtime_environment(self):
    """Verify that the runtime environment is acceptable.

        This method is called as part of __init__ and raises a RuntimeError
        in Python3 or PyPI environments. This module is not compatible with
        Python3 or PyPI. The RuntimeError identifies this to the user up
        front along with suggesting Python 2.6+ be used instead.

        This method also checks that the dependencies qpidtoollibs and
        qpid.messaging are installed. If either one is not installed a
        RuntimeError is raised.

        :raises: RuntimeError if the runtime environment is not acceptable.

        """
    if dependency_is_none(qpidtoollibs):
        raise RuntimeError('The Python package "qpidtoollibs" is missing. Install it with your package manager. You can also try `pip install qpid-tools`.')
    if dependency_is_none(qpid):
        raise RuntimeError('The Python package "qpid.messaging" is missing. Install it with your package manager. You can also try `pip install qpid-python`.')