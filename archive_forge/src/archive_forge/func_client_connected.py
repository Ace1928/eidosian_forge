from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
@abstractmethod
def client_connected(self, telnet_connection):
    """
        Called when a new client was connected.

        Probably you want to call `telnet_connection.set_cli` here to set a
        the CommandLineInterface instance to be used.
        Hint: Use the following shortcut: `prompt_toolkit.shortcuts.create_cli`
        """