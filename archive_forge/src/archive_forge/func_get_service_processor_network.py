from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
import time
def get_service_processor_network(self):
    """
        Return details about service processor network
        :param:
            name : name of the node
        :return: Details about service processor network. None if not found.
        :rtype: dict
        """
    if self.use_rest:
        return self.get_service_processor_network_rest()
    spn_get_iter = netapp_utils.zapi.NaElement('service-processor-network-get-iter')
    query_info = {'query': {'service-processor-network-info': {'node': self.parameters['node']}}}
    spn_get_iter.translate_struct(query_info)
    try:
        result = self.server.invoke_successfully(spn_get_iter, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error fetching service processor network info for %s: %s' % (self.parameters['node'], to_native(error)), exception=traceback.format_exc())
    sp_details = None
    if int(result['num-records']) >= 1:
        sp_details = dict()
        sp_attr_info = result['attributes-list']['service-processor-network-info']
        for item_key, zapi_key in self.na_helper.zapi_string_keys.items():
            sp_details[item_key] = sp_attr_info.get_child_content(zapi_key)
            if item_key == 'dhcp' and sp_details[item_key] is None:
                sp_details[item_key] = 'none'
        for item_key, zapi_key in self.na_helper.zapi_bool_keys.items():
            sp_details[item_key] = self.na_helper.get_value_for_bool(from_zapi=True, value=sp_attr_info.get_child_content(zapi_key))
        for item_key, zapi_key in self.na_helper.zapi_int_keys.items():
            sp_details[item_key] = self.na_helper.get_value_for_int(from_zapi=True, value=sp_attr_info.get_child_content(zapi_key))
    return sp_details