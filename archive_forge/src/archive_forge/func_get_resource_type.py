from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def get_resource_type(self):
    resource_type = self.module.params.get('resource_type')
    return RESOURCE_TYPES.get(resource_type)