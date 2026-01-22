from __future__ import absolute_import, division, print_function
import base64
from .vultr_v2 import AnsibleVultr
def get_resource_vpcs(self, resource, api_version='v1'):
    path = '%s/%s' % (self.resource_path, resource['id'] + self.VPC_CONFIGS[api_version]['path'])
    vpcs = self.query_list(path=path, result_key='vpcs')
    result = list()
    for vpc in vpcs:
        if 'description' in vpc:
            return vpcs
        vpc_detail = self.query_by_id(resource_id=vpc['id'], path=self.VPC_CONFIGS[api_version]['path'], result_key='vpc')
        vpc['description'] = vpc_detail['description']
        result.append(vpc)
    return result