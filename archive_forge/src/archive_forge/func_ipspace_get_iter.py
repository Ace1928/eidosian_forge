from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def ipspace_get_iter(self, name):
    """
        Return net-ipspaces-get-iter query results
        :param name: Name of the ipspace
        :return: NaElement if ipspace found, None otherwise
        """
    ipspace_get_iter = netapp_utils.zapi.NaElement('net-ipspaces-get-iter')
    query_details = netapp_utils.zapi.NaElement.create_node_with_children('net-ipspaces-info', **{'ipspace': name})
    query = netapp_utils.zapi.NaElement('query')
    query.add_child_elem(query_details)
    ipspace_get_iter.add_child_elem(query)
    try:
        result = self.server.invoke_successfully(ipspace_get_iter, enable_tunneling=False)
    except netapp_utils.zapi.NaApiError as error:
        if to_native(error.code) == '14636' or to_native(error.code) == '13073':
            return None
        self.module.fail_json(msg='Error getting ipspace %s: %s' % (name, to_native(error)), exception=traceback.format_exc())
    return result