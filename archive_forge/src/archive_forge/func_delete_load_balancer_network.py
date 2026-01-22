from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.load_balancers import BoundLoadBalancer, PrivateNet
from ..module_utils.vendor.hcloud.networks import BoundNetwork
def delete_load_balancer_network(self):
    self._get_load_balancer_and_network()
    self._get_load_balancer_network()
    if self.hcloud_load_balancer_network is not None and self.hcloud_load_balancer is not None:
        if not self.module.check_mode:
            try:
                self.hcloud_load_balancer.detach_from_network(self.hcloud_load_balancer_network.network).wait_until_finished()
                self._mark_as_changed()
            except HCloudException as exception:
                self.fail_json_hcloud(exception)
    self.hcloud_load_balancer_network = None