import base64
import getpass
import os
import re
import six
import sys
import socket
import threading
from binascii import hexlify
from ncclient.capabilities import Capabilities
from ncclient.logging_ import SessionLoggerAdapter
import paramiko
from ncclient.transport.errors import AuthenticationError, SSHError, SSHUnknownHostError
from ncclient.transport.session import Session
from ncclient.transport.parser import DefaultXMLParser
import logging
def default_unknown_host_cb(host, fingerprint):
    """An unknown host callback returns `True` if it finds the key acceptable, and `False` if not.

    This default callback always returns `False`, which would lead to :meth:`connect` raising a :exc:`SSHUnknownHost` exception.

    Supply another valid callback if you need to verify the host key programmatically.

    *host* is the hostname that needs to be verified

    *fingerprint* is a hex string representing the host key fingerprint, colon-delimited e.g. `"4b:69:6c:72:6f:79:20:77:61:73:20:68:65:72:65:21"`
    """
    return False