from __future__ import absolute_import, division, print_function
from datetime import datetime
import re
from time import sleep
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
def get_unused_pit_key(self):
    """Determine all embedded pit key-values that do not match existing snapshot images."""
    if not self.cache['get_unused_pit_key_values']:
        try:
            rc, images = self.request('storage-systems/%s/snapshot-images' % self.ssid)
            rc, key_values = self.request('key-values')
            for key_value in key_values:
                key = key_value['key']
                value = key_value['value']
                if re.match('ansible\\|.*\\|.*', value):
                    for image in images:
                        if str(image['pitTimestamp']) == value.split('|')[0]:
                            break
                    else:
                        self.cache['get_unused_pit_key_values'].append(key)
        except Exception as error:
            self.module.warn('Failed to retrieve all snapshots to determine all key-value pairs that do no match a point-in-time snapshot images! Array [%s]. Error [%s].' % (self.ssid, error))
    return self.cache['get_unused_pit_key_values']