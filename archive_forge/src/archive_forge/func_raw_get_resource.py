from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.redfish_utils import RedfishUtils
def raw_get_resource(self, resource_uri):
    if resource_uri is None:
        return {'ret': False, 'msg': 'resource_uri is missing'}
    response = self.get_request(self.root_uri + resource_uri)
    if response['ret'] is False:
        return response
    data = response['data']
    return {'ret': True, 'data': data}