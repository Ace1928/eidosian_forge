from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
import codecs
from ansible.module_utils._text import to_text, to_bytes
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_lun(self):
    """
        Return details about the LUN

        :return: Details about the lun
        :rtype: dict
        """
    if self.use_rest:
        return self.get_lun_rest()
    query_details = netapp_utils.zapi.NaElement('lun-info')
    query_details.add_new_child('path', self.parameters['path'])
    query = netapp_utils.zapi.NaElement('query')
    query.add_child_elem(query_details)
    lun_query = netapp_utils.zapi.NaElement('lun-get-iter')
    lun_query.add_child_elem(query)
    result = self.server.invoke_successfully(lun_query, True)
    return_value = None
    if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) >= 1:
        lun = result.get_child_by_name('attributes-list').get_child_by_name('lun-info')
        return_value = {'lun_node': lun.get_child_content('node'), 'lun_ostype': lun.get_child_content('multiprotocol-type'), 'lun_serial': lun.get_child_content('serial-number'), 'lun_naa_id': self.return_naa_id(lun.get_child_content('serial-number')), 'lun_state': lun.get_child_content('state'), 'lun_size': lun.get_child_content('size')}
    return return_value