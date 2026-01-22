from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def register_iso(self):
    args = self._get_common_args()
    args.update({'domainid': self.get_domain('id'), 'account': self.get_account('name'), 'projectid': self.get_project('id'), 'checksum': self.module.params.get('checksum'), 'isfeatured': self.module.params.get('is_featured'), 'ispublic': self.module.params.get('is_public')})
    if not self.module.params.get('cross_zones'):
        args['zoneid'] = self.get_zone(key='id')
    else:
        args['zoneid'] = -1
    if args['bootable'] and (not args['ostypeid']):
        self.module.fail_json(msg="OS type 'os_type' is required if 'bootable=true'.")
    args['url'] = self.module.params.get('url')
    if not args['url']:
        self.module.fail_json(msg='URL is required.')
    self.result['changed'] = True
    if not self.module.check_mode:
        res = self.query_api('registerIso', **args)
        self.iso = res['iso'][0]
    return self.iso