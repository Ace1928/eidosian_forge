from __future__ import absolute_import, division, print_function
from ansible.module_utils.urls import fetch_url, basic_auth_header
from ansible.module_utils.basic import AnsibleModule
def delete_customer(self, id):
    url = '%s/api/customer/%s' % (self.alerta_url, id)
    response = self.send_request(url, None, 'DELETE')
    return response