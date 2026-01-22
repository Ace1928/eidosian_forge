from __future__ import absolute_import, division, print_function
from datetime import datetime
import re
from time import sleep
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
def remove_base_volumes(self, volume_info_list):
    """Add base volume(s) to the consistency group."""
    group_id = self.get_consistency_group()['consistency_group_id']
    for name, info in volume_info_list.items():
        try:
            rc, resp = self.request('storage-systems/%s/consistency-groups/%s/member-volumes/%s' % (self.ssid, group_id, info['id']), method='DELETE')
        except Exception as error:
            self.module.fail_json(msg='Failed to remove reserve capacity volume! Base volume [%s]. Group [%s]. Error [%s]. Array [%s].' % (name, self.group_name, error, self.ssid))