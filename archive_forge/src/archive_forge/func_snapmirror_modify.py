from __future__ import absolute_import, division, print_function
import re
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def snapmirror_modify(self, modify):
    """
        Modify SnapMirror schedule or policy
        """
    if self.use_rest:
        return self.snapmirror_mod_init_resync_break_quiesce_resume_rest(modify=modify)
    options = {'destination-location': self.parameters['destination_path']}
    snapmirror_modify = netapp_utils.zapi.NaElement.create_node_with_children('snapmirror-modify', **options)
    param_to_zapi = {'schedule': 'schedule', 'policy': 'policy', 'max_transfer_rate': 'max-transfer-rate'}
    for param_key, value in modify.items():
        snapmirror_modify.add_new_child(param_to_zapi[param_key], str(value))
    try:
        self.server.invoke_successfully(snapmirror_modify, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error modifying SnapMirror schedule or policy: %s' % to_native(error), exception=traceback.format_exc())