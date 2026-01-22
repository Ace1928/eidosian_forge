from __future__ import absolute_import, division, print_function
import os
import multiprocessing
import threading
from time import sleep
from ansible.module_utils import six
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule, create_multipart_formdata, request
from ansible.module_utils._text import to_native
def is_firmware_bundled(self):
    """Determine whether supplied firmware is bundle."""
    if self.is_bundle_cache is None:
        with open(self.firmware, 'rb') as fh:
            signature = fh.read(16).lower()
            if b'firmware' in signature:
                self.is_bundle_cache = False
            elif b'combined_content' in signature:
                self.is_bundle_cache = True
            else:
                self.module.fail_json(msg='Firmware file is invalid. File [%s]. Array [%s]' % (self.firmware, self.ssid))
    return self.is_bundle_cache