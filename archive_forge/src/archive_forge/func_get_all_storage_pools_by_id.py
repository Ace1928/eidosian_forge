from __future__ import absolute_import, division, print_function
from datetime import datetime
import re
from time import sleep
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
def get_all_storage_pools_by_id(self):
    """Retrieve and return all storage pools/volume groups."""
    if not self.cache['get_all_storage_pools_by_id']:
        try:
            rc, storage_pools = self.request('storage-systems/%s/storage-pools' % self.ssid)
            for storage_pool in storage_pools:
                self.cache['get_all_storage_pools_by_id'].update({storage_pool['id']: storage_pool})
                self.cache['get_all_storage_pools_by_name'].update({storage_pool['name']: storage_pool})
        except Exception as error:
            self.module.fail_json(msg='Failed to retrieve volumes! Error [%s]. Array [%s].' % (error, self.ssid))
    return self.cache['get_all_storage_pools_by_id']