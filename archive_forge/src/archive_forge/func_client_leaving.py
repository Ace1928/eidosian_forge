from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
@abstractmethod
def client_leaving(self, telnet_connection):
    """
        Called when a client quits.
        """