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
def snapmirror_resync(self):
    """
        resync SnapMirror based on relationship state
        """
    if self.use_rest:
        state = 'in_sync' if self.policy_type == 'sync' else 'snapmirrored'
        self.snapmirror_mod_init_resync_break_quiesce_resume_rest(state=state)
    else:
        options = {'destination-location': self.parameters['destination_path']}
        snapmirror_resync = netapp_utils.zapi.NaElement.create_node_with_children('snapmirror-resync', **options)
        try:
            self.server.invoke_successfully(snapmirror_resync, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error resyncing SnapMirror relationship: %s' % to_native(error), exception=traceback.format_exc())
    self.wait_for_idle_status()