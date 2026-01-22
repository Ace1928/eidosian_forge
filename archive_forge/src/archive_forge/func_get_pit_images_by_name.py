from __future__ import absolute_import, division, print_function
from datetime import datetime
import re
from time import sleep
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
def get_pit_images_by_name(self):
    """Retrieve and return snapshot images."""
    if not self.cache['get_pit_images_by_name']:
        self.get_pit_images_by_timestamp()
    return self.cache['get_pit_images_by_name']