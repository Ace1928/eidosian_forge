from __future__ import absolute_import, division, print_function
import json  # noqa: F402
from ansible.module_utils.basic import AnsibleModule  # noqa: F402
from ansible.module_utils._text import to_bytes, to_native  # noqa: F402
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
def update_network_result(self, changed=True):
    """Inspect the current network, update results with last info, exit.

        Keyword Arguments:
            changed {bool} -- whether any action was performed
                              (default: {True})
        """
    facts = self.network.get_info() if changed else self.network.info
    out, err = (self.network.stdout, self.network.stderr)
    self.results.update({'changed': changed, 'network': facts, 'podman_actions': self.network.actions}, stdout=out, stderr=err)
    if self.network.diff:
        self.results.update({'diff': self.network.diff})
    if self.module.params['debug']:
        self.results.update({'podman_version': self.network.version})
    self.module.exit_json(**self.results)