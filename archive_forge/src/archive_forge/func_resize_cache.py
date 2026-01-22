from __future__ import absolute_import, division, print_function
import json
import logging
import sys
import traceback
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves import reduce
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils._text import to_native
from ansible.module_utils.urls import open_url
def resize_cache(self):
    current_disk_count = len(self.cache_detail['driveRefs'])
    proposed_new_disks = 0
    proposed_additional_bytes = 0
    proposed_disk_ids = []
    if self.needs_more_disks:
        proposed_disk_count = self.disk_count - current_disk_count
        disk_ids, bytes = self.get_candidate_disks(disk_count=proposed_disk_count)
        proposed_additional_bytes = bytes
        proposed_disk_ids = disk_ids
        while self.current_size_bytes + proposed_additional_bytes < self.requested_size_bytes:
            proposed_new_disks += 1
            disk_ids, bytes = self.get_candidate_disks(disk_count=proposed_new_disks)
            proposed_disk_ids = disk_ids
            proposed_additional_bytes = bytes
        add_drives_req = dict(driveRef=proposed_disk_ids)
        self.debug('adding drives to flash-cache...')
        rc, self.resp = request(self.api_url + '/storage-systems/%s/flash-cache/addDrives' % self.ssid, data=json.dumps(add_drives_req), headers=self.post_headers, method='POST', url_username=self.api_username, url_password=self.api_password, validate_certs=self.validate_certs)
    elif self.needs_less_disks and self.driveRefs:
        rm_drives = dict(driveRef=self.driveRefs)
        rc, self.resp = request(self.api_url + '/storage-systems/%s/flash-cache/removeDrives' % self.ssid, data=json.dumps(rm_drives), headers=self.post_headers, method='POST', url_username=self.api_username, url_password=self.api_password, validate_certs=self.validate_certs)