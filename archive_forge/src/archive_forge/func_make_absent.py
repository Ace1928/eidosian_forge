from __future__ import (absolute_import, division, print_function)
import json
from ansible.module_utils._text import to_bytes, to_native
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
from ansible_collections.containers.podman.plugins.module_utils.podman.common import generate_systemd
from ansible_collections.containers.podman.plugins.module_utils.podman.common import delete_systemd
def make_absent(self):
    """Run actions if desired state is 'absent'."""
    if not self.pod.exists:
        self.results.update({'changed': False})
    elif self.pod.exists:
        delete_systemd(self.module, self.module_params, self.name, self.pod.version)
        self.pod.delete()
        self.results['actions'].append('deleted %s' % self.pod.name)
        self.results.update({'changed': True})
    self.results.update({'pod': {}, 'podman_actions': self.pod.actions})