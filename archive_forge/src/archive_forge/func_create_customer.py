from __future__ import absolute_import, division, print_function
from ansible.module_utils.urls import fetch_url, basic_auth_header
from ansible.module_utils.basic import AnsibleModule
def create_customer(self):
    url = '%s/api/customer' % self.alerta_url
    payload = {'customer': self.customer, 'match': self.match}
    payload = self.module.jsonify(payload)
    response = self.send_request(url, payload, 'POST')
    return response