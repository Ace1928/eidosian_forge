from __future__ import absolute_import, division, print_function
import json  # noqa: F402
from ansible.module_utils.basic import AnsibleModule  # noqa: F402
from ansible.module_utils._text import to_bytes, to_native  # noqa: F402
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
def diffparam_ipv6(self):
    if LooseVersion(self.version) >= LooseVersion('4.0.0'):
        before = self.info.get('ipv6_enabled', False)
        after = self.params['ipv6']
        return self._diff_update_and_compare('ipv6', before, after)
    before = after = ''
    return self._diff_update_and_compare('ipv6', before, after)