from __future__ import absolute_import, division, print_function
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.rest_application import RestApplication
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_volume
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_luns(self, lun_path=None):
    """
        Return list of LUNs matching vserver and volume names.

        :return: list of LUNs in XML format.
        :rtype: list
        """
    if self.use_rest:
        return self.get_luns_rest(lun_path)
    luns = []
    tag = None
    query_details = netapp_utils.zapi.NaElement('lun-info')
    query_details.add_new_child('vserver', self.parameters['vserver'])
    if lun_path is not None:
        query_details.add_new_child('lun_path', lun_path)
    else:
        query_details.add_new_child('volume', self.parameters['flexvol_name'])
    query = netapp_utils.zapi.NaElement('query')
    query.add_child_elem(query_details)
    while True:
        lun_info = netapp_utils.zapi.NaElement('lun-get-iter')
        lun_info.add_child_elem(query)
        if tag:
            lun_info.add_new_child('tag', tag, True)
        try:
            result = self.server.invoke_successfully(lun_info, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as exc:
            self.module.fail_json(msg='Error fetching luns for %s: %s' % (self.parameters['flexvol_name'] if lun_path is None else lun_path, to_native(exc)), exception=traceback.format_exc())
        if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) >= 1:
            attr_list = result.get_child_by_name('attributes-list')
            luns.extend(attr_list.get_children())
        tag = result.get_child_content('next-tag')
        if tag is None:
            break
    return luns