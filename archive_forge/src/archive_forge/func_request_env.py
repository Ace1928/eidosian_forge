import os
import signal
import struct
import sys
from zope.interface import implementer
from twisted.conch.interfaces import (
from twisted.conch.ssh import channel, common, connection
from twisted.internet import interfaces, protocol
from twisted.logger import Logger
from twisted.python.compat import networkString
def request_env(self, data):
    """
        Process a request to pass an environment variable.

        @param data: The environment variable name and value, each encoded
            as an SSH protocol string and concatenated.
        @type data: L{bytes}
        @return: A true value if the request to pass this environment
            variable was accepted, otherwise a false value.
        """
    if not self.session:
        self.session = ISession(self.avatar)
    if not ISessionSetEnv.providedBy(self.session):
        return 0
    name, value, data = common.getNS(data, 2)
    try:
        self.session.setEnv(name, value)
    except EnvironmentVariableNotPermitted:
        return 0
    except Exception:
        log.failure('Error setting environment variable {name}', name=name)
        return 0
    else:
        return 1