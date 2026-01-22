from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import APIException, HCloudException
from ..module_utils.vendor.hcloud.networks import BoundNetwork
from ..module_utils.vendor.hcloud.servers import BoundServer, PrivateNet
def delete_server_network(self):
    self._get_server_and_network()
    self._get_server_network()
    if self.hcloud_server_network is not None and self.hcloud_server is not None:
        if not self.module.check_mode:
            try:
                self.hcloud_server.detach_from_network(self.hcloud_server_network.network).wait_until_finished()
            except HCloudException as exception:
                self.fail_json_hcloud(exception)
        self._mark_as_changed()
    self.hcloud_server_network = None