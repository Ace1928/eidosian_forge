from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def present_iso(self):
    iso = self.get_iso()
    if not iso:
        iso = self.register_iso()
    else:
        iso = self.update_iso(iso)
    if iso:
        iso = self.ensure_tags(resource=iso, resource_type='ISO')
        self.iso = iso
    return iso