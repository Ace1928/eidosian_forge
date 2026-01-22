from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_zapi_info(self, zapi_get_method, zapi_info, zapi_type=None):
    subsystem_get = netapp_utils.zapi.NaElement(zapi_get_method)
    query = {'query': {zapi_info: {'subsystem': self.parameters.get('subsystem'), 'vserver': self.parameters.get('vserver')}}}
    subsystem_get.translate_struct(query)
    qualifier = ' %s' % zapi_type if zapi_type else ''
    try:
        result = self.server.invoke_successfully(subsystem_get, enable_tunneling=False)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error fetching subsystem%s info: %s' % (qualifier, to_native(error)), exception=traceback.format_exc())
    return result