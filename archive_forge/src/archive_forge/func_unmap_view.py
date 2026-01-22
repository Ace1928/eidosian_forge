from __future__ import absolute_import, division, print_function
from datetime import datetime
import re
from time import sleep
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
def unmap_view(self, unmap_info_list):
    """Unmap consistency group point-in-time snapshot volumes from host or host group."""
    for unmap_info in unmap_info_list:
        try:
            rc, unmap = self.request('storage-systems/%s/volume-mappings/%s' % (self.ssid, unmap_info['lun_mapping_reference']), method='DELETE')
        except Exception as error:
            self.module.fail_json(msg='Failed to unmap snapshot volume! Snapshot volume [%s]. View [%s]. Group [%s]. Array [%s]. Error [%s].' % (unmap_info['snapshot_volume_name'], self.view_name, self.group_name, self.ssid, error))