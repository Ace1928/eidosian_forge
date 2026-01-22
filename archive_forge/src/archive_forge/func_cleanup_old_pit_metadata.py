from __future__ import absolute_import, division, print_function
from datetime import datetime
import re
from time import sleep
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
def cleanup_old_pit_metadata(self, keys):
    """Delete unused point-in-time image metadata."""
    for key in keys:
        try:
            rc, images = self.request('key-values/%s' % key, method='DELETE')
        except Exception as error:
            self.module.fail_json(msg='Failed to purge unused point-in-time image metadata! Key [%s]. Array [%s]. Error [%s].' % (key, self.ssid, error))