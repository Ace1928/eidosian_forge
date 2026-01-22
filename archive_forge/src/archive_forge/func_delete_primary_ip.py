from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.primary_ips import BoundPrimaryIP
def delete_primary_ip(self):
    try:
        self._get_primary_ip()
        if self.hcloud_primary_ip is not None:
            if not self.module.check_mode:
                self.client.primary_ips.delete(self.hcloud_primary_ip)
            self._mark_as_changed()
        self.hcloud_primary_ip = None
    except HCloudException as exception:
        self.fail_json_hcloud(exception)