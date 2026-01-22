from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def get_scanner_pool(self):
    """
        Check to see if a scanner pool exist or not
        :return: True if it exist, False if it does not
        """
    if self.use_rest:
        return self.get_scanner_pool_rest()
    return_value = None
    scanner_pool_obj = netapp_utils.zapi.NaElement('vscan-scanner-pool-get-iter')
    scanner_pool_info = netapp_utils.zapi.NaElement('vscan-scanner-pool-info')
    scanner_pool_info.add_new_child('scanner-pool', self.parameters['scanner_pool'])
    scanner_pool_info.add_new_child('vserver', self.parameters['vserver'])
    query = netapp_utils.zapi.NaElement('query')
    query.add_child_elem(scanner_pool_info)
    scanner_pool_obj.add_child_elem(query)
    try:
        result = self.server.invoke_successfully(scanner_pool_obj, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error searching for Vscan Scanner Pool %s: %s' % (self.parameters['scanner_pool'], to_native(error)), exception=traceback.format_exc())
    if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) >= 1:
        if result.get_child_by_name('attributes-list').get_child_by_name('vscan-scanner-pool-info').get_child_content('scanner-pool') == self.parameters['scanner_pool']:
            scanner_pool_obj = result.get_child_by_name('attributes-list').get_child_by_name('vscan-scanner-pool-info')
            hostname = [host.get_content() for host in scanner_pool_obj.get_child_by_name('hostnames').get_children()]
            privileged_users = [user.get_content() for user in scanner_pool_obj.get_child_by_name('privileged-users').get_children()]
            return_value = {'hostnames': hostname, 'enable': scanner_pool_obj.get_child_content('is-currently-active'), 'privileged_users': privileged_users, 'scanner_pool': scanner_pool_obj.get_child_content('scanner-pool'), 'scanner_policy': scanner_pool_obj.get_child_content('scanner-policy')}
    return return_value