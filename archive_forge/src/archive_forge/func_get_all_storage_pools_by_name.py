from __future__ import absolute_import, division, print_function
from datetime import datetime
import re
from time import sleep
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
def get_all_storage_pools_by_name(self):
    """Retrieve and return all storage pools/volume groups."""
    if not self.cache['get_all_storage_pools_by_name']:
        self.get_all_storage_pools_by_id()
    return self.cache['get_all_storage_pools_by_name']