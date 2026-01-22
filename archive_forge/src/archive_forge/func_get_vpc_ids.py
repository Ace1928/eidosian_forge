from __future__ import absolute_import, division, print_function
import base64
from .vultr_v2 import AnsibleVultr
def get_vpc_ids(self, api_version='v1'):
    vpc_names = list(self.module.params[self.VPC_CONFIGS[api_version]['param']])
    vpcs = self.query_list(self.VPC_CONFIGS[api_version]['path'], result_key='vpcs')
    vpc_ids = list()
    for vpc in vpcs:
        if self.module.params['region'] != vpc['region']:
            continue
        if vpc['description'] in vpc_names:
            vpc_ids.append(vpc['id'])
            vpc_names.remove(vpc['description'])
    if vpc_names:
        self.module.fail_json(msg='VPCs (%s) not found: %s' % (api_version, ', '.join(vpc_names)))
    return vpc_ids