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
def resize_lun(self, path):
    """
        Resize requested LUN

        :return: True if LUN was actually re-sized, false otherwise.
        :rtype: bool
        """
    if self.use_rest:
        return self.resize_lun_rest()
    lun_resize = netapp_utils.zapi.NaElement.create_node_with_children('lun-resize', **{'path': path, 'size': str(self.parameters['size']), 'force': str(self.parameters['force_resize'])})
    try:
        self.server.invoke_successfully(lun_resize, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as exc:
        if to_native(exc.code) == '9042':
            return False
        else:
            self.module.fail_json(msg='Error resizing lun %s: %s' % (path, to_native(exc)), exception=traceback.format_exc())
    return True