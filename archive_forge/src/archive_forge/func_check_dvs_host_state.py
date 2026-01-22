from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible.module_utils._text import to_native
def check_dvs_host_state(self):
    if self.uplink_portgroup is None:
        self.module.fail_json(msg='An uplink portgroup does not exist on the distributed virtual switch %s' % self.switch_name)
    self.host = self.find_host_attached_dvs()
    if self.host is None:
        self.host = find_hostsystem_by_name(self.content, self.esxi_hostname)
        if self.host is None:
            self.module.fail_json(msg='The esxi_hostname %s does not exist in vCenter' % self.esxi_hostname)
        return 'absent'
    elif self.state == 'absent':
        return 'present'
    elif self.check_uplinks():
        return 'present'
    else:
        return 'update'