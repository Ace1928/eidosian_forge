from __future__ import absolute_import, division, print_function
import json
import os
import uuid
from ansible.module_utils.urls import open_url
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.six.moves.urllib.parse import urlparse
def upload_firmware_image(self, update_image_path):
    """Perform Firmware Upload to the OCAPI storage device.

        :param str update_image_path: The path/filename of the firmware image, on the local filesystem.
        """
    if not (os.path.exists(update_image_path) and os.path.isfile(update_image_path)):
        return {'ret': False, 'msg': 'File does not exist.'}
    url = self.root_uri + 'OperatingSystem'
    url = self.get_uri_with_slot_number_query_param(url)
    content_type, b_form_data = self.prepare_multipart_firmware_upload(update_image_path)
    if self.module.check_mode:
        return {'ret': True, 'changed': True, 'msg': 'Update not performed in check mode.'}
    result = self.post_request(url, b_form_data, content_type=content_type, timeout=300)
    if result['ret'] is False:
        return result
    return {'ret': True}