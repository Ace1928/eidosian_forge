from __future__ import absolute_import, division, print_function
import functools
from itertools import groupby
from time import sleep
from pprint import pformat
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import NetAppESeriesModule
from ansible.module_utils._text import to_native
def expand_storage_pool(self, check_mode=False):
    """Add drives to existing storage pool.

        :return bool: whether drives were required to be added to satisfy the specified criteria."""
    expansion_candidate_list = self.get_expansion_candidate_drives()
    changed_required = bool(expansion_candidate_list)
    estimated_completion_time = 0.0
    required_expansion_candidate_list = list()
    while expansion_candidate_list:
        subset = list()
        while expansion_candidate_list and len(subset) < self.expandable_drive_count:
            subset.extend(expansion_candidate_list.pop()['drives'])
        required_expansion_candidate_list.append(subset)
    if required_expansion_candidate_list and (not check_mode):
        url = 'storage-systems/%s/symbol/startVolumeGroupExpansion?verboseErrorResponse=true' % self.ssid
        if self.raid_level == 'raidDiskPool':
            url = 'storage-systems/%s/symbol/startDiskPoolExpansion?verboseErrorResponse=true' % self.ssid
        while required_expansion_candidate_list:
            candidate_drives_list = required_expansion_candidate_list.pop()
            request_body = dict(volumeGroupRef=self.pool_detail['volumeGroupRef'], driveRef=candidate_drives_list)
            try:
                rc, resp = self.request(url, method='POST', data=request_body)
            except Exception as error:
                rc, actions_resp = self.request('storage-systems/%s/storage-pools/%s/action-progress' % (self.ssid, self.pool_detail['id']), ignore_errors=True)
                if rc == 200 and actions_resp:
                    actions = [action['currentAction'] for action in actions_resp if action['volumeRef'] in self.storage_pool_volumes]
                    self.module.fail_json(msg='Failed to add drives to the storage pool possibly because of actions in progress. Actions [%s]. Pool id [%s]. Array id [%s]. Error[%s].' % (', '.join(actions), self.pool_detail['id'], self.ssid, to_native(error)))
                self.module.fail_json(msg='Failed to add drives to storage pool. Pool id [%s]. Array id [%s].  Error[%s].' % (self.pool_detail['id'], self.ssid, to_native(error)))
            if required_expansion_candidate_list:
                for dummy in range(self.EXPANSION_TIMEOUT_SEC):
                    rc, actions_resp = self.request('storage-systems/%s/storage-pools/%s/action-progress' % (self.ssid, self.pool_detail['id']), ignore_errors=True)
                    if rc == 200:
                        for action in actions_resp:
                            if action['volumeRef'] in self.storage_pool_volumes and action['currentAction'] == 'remappingDce':
                                sleep(1)
                                estimated_completion_time = action['estimatedTimeToCompletion']
                                break
                        else:
                            estimated_completion_time = 0.0
                            break
    return (changed_required, estimated_completion_time)