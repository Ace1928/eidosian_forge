from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def present_vpc(self):
    vpc = self.get_vpc()
    if not vpc:
        vpc = self._create_vpc(vpc)
    else:
        vpc = self._update_vpc(vpc)
    if vpc:
        vpc = self.ensure_tags(resource=vpc, resource_type='Vpc')
    return vpc