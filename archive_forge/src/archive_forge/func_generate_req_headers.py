from __future__ import absolute_import, division, print_function
import xml.etree.ElementTree as ET
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
def generate_req_headers(self, url):
    req = vim.SessionManager.HttpServiceRequestSpec(method='httpGet', url=url)
    ticket = self.content.sessionManager.AcquireGenericServiceTicket(req)
    headers = {'Content-Type': 'application/octet-stream', 'Cookie': 'vmware_cgi_ticket=%s' % ticket.id}
    return headers