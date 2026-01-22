from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def destroy_dns(self, dns_attrs):
    """
        Destroys an already created dns
        :return:
        """
    if self.use_rest:
        return self.destroy_dns_rest(dns_attrs)
    try:
        self.server.invoke_successfully(netapp_utils.zapi.NaElement('net-dns-destroy'), True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error destroying dns: %s' % to_native(error), exception=traceback.format_exc())