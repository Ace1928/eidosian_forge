from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.isos import BoundIso
def get_iso_infos(self):
    try:
        if self.module.params.get('id') is not None:
            self.hcloud_iso_info = [self.client.isos.get_by_id(self.module.params.get('id'))]
        elif self.module.params.get('name') is not None:
            self.hcloud_iso_info = [self.client.isos.get_by_name(self.module.params.get('name'))]
        else:
            self.hcloud_iso_info = self.client.isos.get_all(architecture=self.module.params.get('architecture'), include_wildcard_architecture=self.module.params.get('include_wildcard_architecture'))
    except HCloudException as exception:
        self.fail_json_hcloud(exception)