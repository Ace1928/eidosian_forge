from __future__ import (absolute_import, division, print_function)
import json
from ansible.module_utils._text import to_bytes, to_native
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
from ansible_collections.containers.podman.plugins.module_utils.podman.common import generate_systemd
from ansible_collections.containers.podman.plugins.module_utils.podman.common import delete_systemd
def construct_command_from_params(self):
    """Create a podman command from given module parameters.

        Returns:
           list -- list of byte strings for Popen command
        """
    if self.action in ['start', 'restart', 'stop', 'delete', 'pause', 'unpause', 'kill']:
        return self._simple_action()
    if self.action in ['create']:
        return self._create_action()
    self.module.fail_json(msg='Unknown action %s' % self.action)