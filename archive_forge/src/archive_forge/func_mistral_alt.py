import os
import os_client_config
from tempest.lib.cli import base
def mistral_alt(self, action, flags='', params='', mode='alt_user'):
    """Executes Mistral command for alt_user from alt_tenant."""
    mistral_url_op = '--os-mistral-url %s' % self._mistral_url
    flags = '{} --insecure'.format(flags)
    return self.clients.cmd_with_auth('mistral %s' % mistral_url_op, action, flags, params)