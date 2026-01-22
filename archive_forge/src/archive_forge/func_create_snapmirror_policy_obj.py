from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def create_snapmirror_policy_obj(self, snapmirror_policy_obj):
    if 'comment' in self.parameters.keys():
        snapmirror_policy_obj.add_new_child('comment', self.parameters['comment'])
    if 'common_snapshot_schedule' in self.parameters.keys() and self.parameters['policy_type'] in ('sync_mirror', 'strict_sync_mirror'):
        snapmirror_policy_obj.add_new_child('common-snapshot-schedule', self.parameters['common_snapshot_schedule'])
    if 'ignore_atime' in self.parameters.keys():
        snapmirror_policy_obj.add_new_child('ignore-atime', self.na_helper.get_value_for_bool(False, self.parameters['ignore_atime']))
    if 'is_network_compression_enabled' in self.parameters.keys():
        snapmirror_policy_obj.add_new_child('is-network-compression-enabled', self.na_helper.get_value_for_bool(False, self.parameters['is_network_compression_enabled']))
    if 'owner' in self.parameters.keys():
        snapmirror_policy_obj.add_new_child('owner', self.parameters['owner'])
    if 'restart' in self.parameters.keys():
        snapmirror_policy_obj.add_new_child('restart', self.parameters['restart'])
    if 'transfer_priority' in self.parameters.keys():
        snapmirror_policy_obj.add_new_child('transfer-priority', self.parameters['transfer_priority'])
    if 'tries' in self.parameters.keys():
        snapmirror_policy_obj.add_new_child('tries', self.parameters['tries'])
    return snapmirror_policy_obj