from __future__ import absolute_import, division, print_function
from datetime import datetime
import re
from time import sleep
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
def get_all_volumes_by_id(self):
    """Retrieve and return a dictionary of all thick and thin volumes keyed by id."""
    if not self.cache['get_all_volumes_by_id']:
        try:
            rc, thick_volumes = self.request('storage-systems/%s/volumes' % self.ssid)
            rc, thin_volumes = self.request('storage-systems/%s/thin-volumes' % self.ssid)
            for volume in thick_volumes + thin_volumes:
                self.cache['get_all_volumes_by_id'].update({volume['id']: volume})
                self.cache['get_all_volumes_by_name'].update({volume['name']: volume})
        except Exception as error:
            self.module.fail_json(msg='Failed to retrieve volumes! Error [%s]. Array [%s].' % (error, self.ssid))
    return self.cache['get_all_volumes_by_id']