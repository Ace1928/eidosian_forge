from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def get_interval_type(self):
    interval_type = self.module.params.get('interval_type')
    return self.interval_types[interval_type]