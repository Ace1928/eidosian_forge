from __future__ import absolute_import, division, print_function
from datetime import datetime
import re
from time import sleep
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
def get_pit_images_by_timestamp(self):
    """Retrieve and return snapshot images."""
    if not self.cache['get_pit_images_by_timestamp']:
        group_id = self.get_consistency_group()['consistency_group_id']
        images_metadata = self.get_pit_images_metadata()
        existing_volumes_by_id = self.get_all_volumes_by_id()
        try:
            rc, images = self.request('storage-systems/%s/consistency-groups/%s/snapshots' % (self.ssid, group_id))
            for image_info in images:
                metadata = {'id': '', 'name': '', 'description': ''}
                if image_info['pitTimestamp'] in images_metadata.keys():
                    metadata = images_metadata[image_info['pitTimestamp']]
                timestamp = datetime.fromtimestamp(int(image_info['pitTimestamp']))
                info = {'id': image_info['id'], 'name': metadata['name'], 'timestamp': timestamp, 'description': metadata['description'], 'sequence_number': image_info['pitSequenceNumber'], 'base_volume_id': image_info['baseVol'], 'base_volume_name': existing_volumes_by_id[image_info['baseVol']]['name'], 'image_info': image_info}
                if timestamp not in self.cache['get_pit_images_by_timestamp'].keys():
                    self.cache['get_pit_images_by_timestamp'].update({timestamp: {'sequence_number': image_info['pitSequenceNumber'], 'images': [info]}})
                    if metadata['name']:
                        self.cache['get_pit_images_by_name'].update({metadata['name']: {'sequence_number': image_info['pitSequenceNumber'], 'images': [info]}})
                else:
                    self.cache['get_pit_images_by_timestamp'][timestamp]['images'].append(info)
                    if metadata['name']:
                        self.cache['get_pit_images_by_name'][metadata['name']]['images'].append(info)
        except Exception as error:
            self.module.fail_json(msg='Failed to retrieve consistency group snapshot images! Group [%s]. Array [%s]. Error [%s].' % (self.group_name, self.ssid, error))
    return self.cache['get_pit_images_by_timestamp']