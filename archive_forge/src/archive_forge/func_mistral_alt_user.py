import os
import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
def mistral_alt_user(self, cmd, params=''):
    self.clients = self._get_alt_clients()
    return self.parser.listing(self.mistral_alt('{0}'.format(cmd), params='{0}'.format(params)))