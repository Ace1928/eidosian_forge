from collections import namedtuple
import logging
import sys
from cliff import app
from cliff import command
from cliff import commandmanager
from cliff import complete
from cliff import help
from keystoneauth1 import identity
from keystoneauth1 import loading
from keystoneauth1 import session
import barbicanclient
from barbicanclient._i18n import _LW
from barbicanclient import client
def prepare_to_run_command(self, cmd):
    """Prepares to run the command

        Checks if the minimal parameters are provided and creates the
        client interface.
        This is inherited from the framework.
        """
    self.client_manager = namedtuple('ClientManager', 'key_manager')
    if cmd.auth_required:
        self.configure_logging()
        self.client_manager.key_manager = self.create_client(self.options)