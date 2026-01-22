from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def create_vpc_offering(self):
    vpc_offering = None
    self.result['changed'] = True
    args = {'name': self.module.params.get('name'), 'state': self.module.params.get('state'), 'displaytext': self.module.params.get('display_text'), 'supportedservices': self.module.params.get('supported_services'), 'serviceproviderlist': self.module.params.get('service_providers'), 'serviceofferingid': self.get_service_offering_id(), 'servicecapabilitylist': self.module.params.get('service_capabilities')}
    required_params = ['display_text', 'supported_services']
    self.module.fail_on_missing_params(required_params=required_params)
    if not self.module.check_mode:
        res = self.query_api('createVPCOffering', **args)
        poll_async = self.module.params.get('poll_async')
        if poll_async:
            vpc_offering = self.poll_job(res, 'vpcoffering')
    return vpc_offering