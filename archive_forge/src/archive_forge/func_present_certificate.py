from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.certificates import BoundCertificate
def present_certificate(self):
    self._get_certificate()
    if self.hcloud_certificate is None:
        self._create_certificate()
    else:
        self._update_certificate()