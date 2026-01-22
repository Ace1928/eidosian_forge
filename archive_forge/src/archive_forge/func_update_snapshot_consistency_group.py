from __future__ import absolute_import, division, print_function
from datetime import datetime
import re
from time import sleep
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
def update_snapshot_consistency_group(self, group_info):
    """Create a new snapshot consistency group."""
    group_id = self.get_consistency_group()['consistency_group_id']
    consistency_group_request = {'name': self.group_name}
    if 'alert_threshold_pct' in group_info.keys():
        consistency_group_request.update({'fullWarnThresholdPercent': group_info['alert_threshold_pct']})
    if 'maximum_snapshots' in group_info.keys():
        consistency_group_request.update({'autoDeleteThreshold': group_info['maximum_snapshots']})
    if 'reserve_capacity_full_policy' in group_info.keys():
        consistency_group_request.update({'repositoryFullPolicy': group_info['reserve_capacity_full_policy']})
    if 'rollback_priority' in group_info.keys():
        consistency_group_request.update({'rollbackPriority': group_info['rollback_priority']})
    try:
        rc, group = self.request('storage-systems/%s/consistency-groups/%s' % (self.ssid, group_id), method='POST', data=consistency_group_request)
        return group['cgRef']
    except Exception as error:
        self.module.fail_json(msg='Failed to remove snapshot consistency group! Group [%s]. Array [%s].' % (self.group_name, self.ssid))