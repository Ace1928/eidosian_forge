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
def snapmirror_mod_init_resync_break_quiesce_resume_rest(self, state=None, modify=None, before_delete=False):
    """
        To perform SnapMirror modify, init, resume, resync and break.
        1. Modify only update SnapMirror policy which passes the policy in body.
        2. To perform SnapMirror init - state=snapmirrored and mirror_state=uninitialized.
        3. To perform SnapMirror resync - state=snapmirrored and mirror_state=broken_off.
        4. To perform SnapMirror break -  state=broken_off and transfer_state not transferring.
        5. To perform SnapMirror quiesce - state=pause and mirror_state not broken_off.
        6. To perform SnapMirror resume - state=snapmirrored.
        """
    uuid = self.get_relationship_uuid()
    if uuid is None:
        self.module.fail_json(msg='Error in updating SnapMirror relationship: unable to get UUID for the SnapMirror relationship.')
    body = {}
    if state is not None:
        body['state'] = state
    elif modify:
        for key in modify:
            if key == 'policy':
                body[key] = {'name': modify[key]}
            elif key == 'schedule':
                body['transfer_schedule'] = {'name': self.string_or_none(modify[key])}
            else:
                self.module.warn(msg='Unexpected key in modify: %s, value: %s' % (key, modify[key]))
    else:
        self.na_helper.changed = False
        return
    api = 'snapmirror/relationships'
    dummy, error = rest_generic.patch_async(self.rest_api, api, uuid, body)
    if error:
        msg = 'Error patching SnapMirror: %s: %s' % (body, to_native(error))
        if before_delete:
            self.previous_errors.append(msg)
        else:
            self.module.fail_json(msg=msg, exception=traceback.format_exc())