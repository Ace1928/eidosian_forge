from __future__ import absolute_import, division, print_function
from datetime import datetime
import re
from time import sleep
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
def create_view_changes_required(self):
    """Determine whether snapshot consistency group point-in-time view needs to be created."""
    changes = {}
    snapshot_images_info = self.get_pit_info()
    changes.update({'name': self.view_name, 'sequence_number': snapshot_images_info['sequence_number'], 'images': snapshot_images_info['images'], 'volumes': self.volumes})
    return changes