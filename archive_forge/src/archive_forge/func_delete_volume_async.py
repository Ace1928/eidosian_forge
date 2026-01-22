from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.rest_application import RestApplication
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def delete_volume_async(self, current):
    """Delete ONTAP volume for infinite or flexgroup types """
    errors = None
    if current['is_online']:
        dummy, errors = self.change_volume_state(call_from_delete_vol=True)
    volume_delete = netapp_utils.zapi.NaElement.create_node_with_children('volume-destroy-async', **{'volume-name': self.parameters['name']})
    try:
        result = self.server.invoke_successfully(volume_delete, enable_tunneling=True)
        self.check_invoke_result(result, 'delete')
    except netapp_utils.zapi.NaApiError as error:
        msg = 'Error deleting volume %s: %s.' % (self.parameters['name'], to_native(error))
        if errors:
            msg += '  Previous errors when offlining/unmounting volume: %s' % ' - '.join(errors)
        self.module.fail_json(msg=msg)