from __future__ import absolute_import, division, print_function
import os
import multiprocessing
import threading
from time import sleep
from ansible.module_utils import six
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule, create_multipart_formdata, request
from ansible.module_utils._text import to_native
def embedded_firmware_activate(self):
    """Activate firmware."""
    rc, response = self.request('firmware/embedded-firmware/activate', method='POST', ignore_errors=True, timeout=10)
    if rc == '422':
        self.module.fail_json(msg='Failed to activate the staged firmware. Array Id [%s]. Error [%s]' % (self.ssid, response))