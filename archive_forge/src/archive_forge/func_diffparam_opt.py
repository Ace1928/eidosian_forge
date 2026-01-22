from __future__ import absolute_import, division, print_function
import json  # noqa: F402
from ansible.module_utils.basic import AnsibleModule  # noqa: F402
from ansible.module_utils._text import to_bytes, to_native  # noqa: F402
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
def diffparam_opt(self):
    if LooseVersion(self.version) >= LooseVersion('4.0.0'):
        vlan_before = self.info.get('options', {}).get('vlan')
    else:
        try:
            vlan_before = self.info['plugins'][0].get('vlan')
        except (IndexError, KeyError):
            vlan_before = None
    vlan_after = self.params['opt'].get('vlan') if self.params['opt'] else None
    if vlan_before or vlan_after:
        before, after = ({'vlan': str(vlan_before)}, {'vlan': str(vlan_after)})
    else:
        before, after = ({}, {})
    if LooseVersion(self.version) >= LooseVersion('4.0.0'):
        mtu_before = self.info.get('options', {}).get('mtu')
    else:
        try:
            mtu_before = self.info['plugins'][0].get('mtu')
        except (IndexError, KeyError):
            mtu_before = None
    mtu_after = self.params['opt'].get('mtu') if self.params['opt'] else None
    if mtu_before or mtu_after:
        before.update({'mtu': str(mtu_before)})
        after.update({'mtu': str(mtu_after)})
    return self._diff_update_and_compare('opt', before, after)