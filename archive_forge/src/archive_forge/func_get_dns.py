from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def get_dns(self):
    if self.use_rest:
        return self.get_dns_rest()
    dns_obj = netapp_utils.zapi.NaElement('net-dns-get')
    try:
        result = self.server.invoke_successfully(dns_obj, True)
    except netapp_utils.zapi.NaApiError as error:
        if to_native(error.code) == '15661':
            return None
        else:
            self.module.fail_json(msg='Error getting DNS info: %s.' % to_native(error), exception=traceback.format_exc())
    attributes = result.get_child_by_name('attributes')
    if attributes is None:
        return
    dns_info = attributes.get_child_by_name('net-dns-info')
    nameservers = dns_info.get_child_by_name('name-servers')
    attrs = {'nameservers': [each.get_content() for each in nameservers.get_children()]}
    domains = dns_info.get_child_by_name('domains')
    attrs['domains'] = [each.get_content() for each in domains.get_children()]
    return attrs