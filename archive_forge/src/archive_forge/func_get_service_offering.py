from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def get_service_offering(self):
    args = {'name': self.module.params.get('name'), 'domainid': self.get_domain(key='id'), 'issystem': self.module.params.get('is_system'), 'systemvmtype': self.module.params.get('system_vm_type')}
    service_offerings = self.query_api('listServiceOfferings', **args)
    if service_offerings:
        return service_offerings['serviceoffering'][0]