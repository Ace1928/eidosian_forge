from __future__ import absolute_import, division, print_function
from ..module_utils.cloudstack import AnsibleCloudStack, cs_argument_spec, cs_required_together
from ansible.module_utils.basic import AnsibleModule
def present_traffic_type(self):
    traffic_type = self.get_traffic_type()
    if traffic_type:
        self.traffic_type = self.update_traffic_type()
    else:
        self.result['changed'] = True
        self.traffic_type = self.add_traffic_type()
    return self.traffic_type