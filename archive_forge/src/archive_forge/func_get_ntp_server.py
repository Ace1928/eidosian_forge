from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_ntp_server(self):
    """
        Return details about the ntp server
        :param:
            name : Name of the server_name
        :return: Details about the ntp server. None if not found.
        :rtype: dict
        """
    if self.use_rest:
        return self.get_ntp_server_rest()
    ntp_iter = netapp_utils.zapi.NaElement('ntp-server-get-iter')
    ntp_info = netapp_utils.zapi.NaElement('ntp-server-info')
    ntp_info.add_new_child('server-name', self.parameters['server_name'])
    query = netapp_utils.zapi.NaElement('query')
    query.add_child_elem(ntp_info)
    ntp_iter.add_child_elem(query)
    result = self.server.invoke_successfully(ntp_iter, True)
    return_value = None
    if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) == 1:
        ntp_server_name = result.get_child_by_name('attributes-list').get_child_by_name('ntp-server-info').get_child_content('server-name')
        server_version = result.get_child_by_name('attributes-list').get_child_by_name('ntp-server-info').get_child_content('version')
        server_key_id = result.get_child_by_name('attributes-list').get_child_by_name('ntp-server-info').get_child_content('key-id')
        return_value = {'server-name': ntp_server_name, 'version': server_version, 'key_id': int(server_key_id) if server_key_id is not None else 0}
    return return_value