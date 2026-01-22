from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def get_nic(self):
    if self.nic:
        return self.nic
    args = {'virtualmachineid': self.get_vm(key='id'), 'networkid': self.get_network(key='id')}
    nics = self.query_api('listNics', **args)
    if nics:
        self.nic = nics['nic'][0]
        return self.nic
    self.fail_json(msg='NIC for VM %s in network %s not found' % (self.get_vm(key='name'), self.get_network(key='name')))