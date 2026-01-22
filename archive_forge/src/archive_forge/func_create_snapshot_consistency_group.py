from __future__ import absolute_import, division, print_function
from datetime import datetime
import re
from time import sleep
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
def create_snapshot_consistency_group(self, group_info):
    """Create a new snapshot consistency group."""
    consistency_group_request = {'name': self.group_name, 'fullWarnThresholdPercent': group_info['alert_threshold_pct'], 'autoDeleteThreshold': group_info['maximum_snapshots'], 'repositoryFullPolicy': group_info['reserve_capacity_full_policy'], 'rollbackPriority': group_info['rollback_priority']}
    try:
        rc, group = self.request('storage-systems/%s/consistency-groups' % self.ssid, method='POST', data=consistency_group_request)
        self.cache['get_consistency_group'].update({'consistency_group_id': group['cgRef']})
    except Exception as error:
        self.module.fail_json(msg='Failed to remove snapshot consistency group! Group [%s]. Array [%s].' % (self.group_name, self.ssid))