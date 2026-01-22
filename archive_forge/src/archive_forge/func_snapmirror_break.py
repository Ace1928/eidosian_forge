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
def snapmirror_break(self, destination=None, before_delete=False):
    """
        Break SnapMirror relationship at destination cluster
        #1. Quiesce the SnapMirror relationship at destination
        #2. Break the SnapMirror relationship at the destination
        """
    self.snapmirror_quiesce()
    if self.use_rest:
        if self.parameters['current_mirror_state'] == 'broken_off' or self.parameters['current_transfer_status'] == 'transferring':
            self.na_helper.changed = False
            self.module.fail_json(msg='snapmirror data are transferring')
        return self.snapmirror_mod_init_resync_break_quiesce_resume_rest(state='broken_off', before_delete=before_delete)
    if destination is None:
        destination = self.parameters['destination_path']
    options = {'destination-location': destination}
    snapmirror_break = netapp_utils.zapi.NaElement.create_node_with_children('snapmirror-break', **options)
    try:
        self.server.invoke_successfully(snapmirror_break, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        msg = 'Error breaking SnapMirror relationship: %s' % to_native(error)
        if before_delete:
            self.previous_errors.append(msg)
        else:
            self.module.fail_json(msg=msg, exception=traceback.format_exc())