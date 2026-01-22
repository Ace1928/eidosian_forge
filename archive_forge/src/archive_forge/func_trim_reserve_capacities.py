from __future__ import absolute_import, division, print_function
from datetime import datetime
import re
from time import sleep
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
def trim_reserve_capacities(self, trim_reserve_volume_info_list):
    """trim base volume(s) reserve capacity."""
    for info in trim_reserve_volume_info_list:
        trim_request = {'concatVol': info['concat_volume_id'], 'trimCount': info['trim_count'], 'retainRepositoryMembers': False}
        try:
            rc, trim = self.request('storage-systems/%s/symbol/trimConcatVolume?verboseErrorResponse=true' % self.ssid, method='POST', data=trim_request)
        except Exception as error:
            self.module.fail_json(msg='Failed to trim reserve capacity. Group [%s]. Array [%s]. Error [%s].' % (self.group_name, self.ssid, error))