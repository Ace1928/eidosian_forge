from __future__ import absolute_import, division, print_function
import base64
from .vultr_v2 import AnsibleVultr
def get_user_data(self, resource):
    res = self.api_query(path='%s/%s/%s' % (self.resource_path, resource[self.resource_key_id], 'user-data'))
    if res:
        return str(res.get('user_data', dict()).get('data'))
    return ''