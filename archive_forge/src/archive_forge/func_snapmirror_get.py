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
def snapmirror_get(self, destination=None):
    """
        Get current SnapMirror relations
        :return: Dictionary of current SnapMirror details if query successful, else None
        """
    if self.use_rest:
        return self.snapmirror_get_rest(destination)
    snapmirror_get_iter = self.snapmirror_get_iter(destination)
    try:
        result = self.server.invoke_successfully(snapmirror_get_iter, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error fetching snapmirror info: %s' % to_native(error), exception=traceback.format_exc())
    if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) > 0:
        snapmirror_info = result.get_child_by_name('attributes-list').get_child_by_name('snapmirror-info')
        snap_info = {}
        snap_info['mirror_state'] = snapmirror_info.get_child_content('mirror-state')
        snap_info['status'] = snapmirror_info.get_child_content('relationship-status')
        snap_info['schedule'] = snapmirror_info.get_child_content('schedule')
        snap_info['policy'] = snapmirror_info.get_child_content('policy')
        snap_info['relationship_type'] = snapmirror_info.get_child_content('relationship-type')
        snap_info['current_transfer_type'] = snapmirror_info.get_child_content('current-transfer-type')
        snap_info['source_path'] = snapmirror_info.get_child_content('source-location')
        if snapmirror_info.get_child_by_name('max-transfer-rate'):
            snap_info['max_transfer_rate'] = int(snapmirror_info.get_child_content('max-transfer-rate'))
        if snapmirror_info.get_child_by_name('last-transfer-error'):
            snap_info['last_transfer_error'] = snapmirror_info.get_child_content('last-transfer-error')
        if snapmirror_info.get_child_by_name('is-healthy') is not None:
            snap_info['is_healthy'] = self.na_helper.get_value_for_bool(True, snapmirror_info.get_child_content('is-healthy'))
        if snapmirror_info.get_child_by_name('unhealthy-reason'):
            snap_info['unhealthy_reason'] = snapmirror_info.get_child_content('unhealthy-reason')
        if snap_info['schedule'] is None:
            snap_info['schedule'] = ''
        return snap_info
    return None