import importlib
import logging
import sys
from osc_lib import clientmanager
from osc_lib import shell
import stevedore
def setup_auth(self):
    """Set up authentication"""
    if self._auth_setup_completed:
        return
    if self._auth_required and self._cli_options._openstack_config is not None:
        self._cli_options._openstack_config._pw_callback = shell.prompt_for_password
        try:
            if not self._cli_options._auth:
                self._cli_options._auth = self._cli_options._openstack_config.load_auth_plugin(self._cli_options.config)
        except TypeError as e:
            self._fallback_load_auth_plugin(e)
    return super(ClientManager, self).setup_auth()