from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.primary_ips import BoundPrimaryIP
def get_primary_ips(self):
    try:
        if self.module.params.get('id') is not None:
            self.hcloud_primary_ip_info = [self.client.primary_ips.get_by_id(self.module.params.get('id'))]
        elif self.module.params.get('name') is not None:
            self.hcloud_primary_ip_info = [self.client.primary_ips.get_by_name(self.module.params.get('name'))]
        elif self.module.params.get('label_selector') is not None:
            self.hcloud_primary_ip_info = self.client.primary_ips.get_all(label_selector=self.module.params.get('label_selector'))
        else:
            self.hcloud_primary_ip_info = self.client.primary_ips.get_all()
    except HCloudException as exception:
        self.fail_json_hcloud(exception)