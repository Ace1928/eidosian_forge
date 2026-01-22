from __future__ import absolute_import, division, print_function
import re
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def attach_object_store_to_aggr(self):
    """
        Attach object store to aggregate.
        :return: None
        """
    if self.use_rest:
        return self.attach_object_store_to_aggr_rest()
    store_obj = {'aggregate': self.parameters['name'], 'object-store-name': self.parameters['object_store_name']}
    if 'allow_flexgroups' in self.parameters:
        store_obj['allow-flexgroup'] = self.na_helper.get_value_for_bool(False, self.parameters['allow_flexgroups'])
    attach_object_store = netapp_utils.zapi.NaElement.create_node_with_children('aggr-object-store-attach', **store_obj)
    try:
        self.server.invoke_successfully(attach_object_store, enable_tunneling=False)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error attaching object store %s to aggregate %s: %s' % (self.parameters['object_store_name'], self.parameters['name'], to_native(error)), exception=traceback.format_exc())