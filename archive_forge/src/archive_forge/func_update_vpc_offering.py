from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def update_vpc_offering(self, vpc_offering):
    if not vpc_offering:
        return vpc_offering
    args = {'id': vpc_offering['id'], 'state': self.module.params.get('state'), 'name': self.module.params.get('name'), 'displaytext': self.module.params.get('display_text')}
    if args['state'] in ['enabled', 'disabled']:
        args['state'] = args['state'].title()
    else:
        del args['state']
    if self.has_changed(args, vpc_offering):
        self.result['changed'] = True
        if not self.module.check_mode:
            res = self.query_api('updateVPCOffering', **args)
            poll_async = self.module.params.get('poll_async')
            if poll_async:
                vpc_offering = self.poll_job(res, 'vpcoffering')
    return vpc_offering