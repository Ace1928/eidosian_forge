from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.networks import BoundNetwork, NetworkSubnet
def present_subnetwork(self):
    self._get_network()
    self._get_subnetwork()
    if self.hcloud_subnetwork is None:
        self._create_subnetwork()