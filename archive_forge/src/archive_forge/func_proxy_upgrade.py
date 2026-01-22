from __future__ import absolute_import, division, print_function
import os
import multiprocessing
import threading
from time import sleep
from ansible.module_utils import six
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule, create_multipart_formdata, request
from ansible.module_utils._text import to_native
def proxy_upgrade(self):
    """Activate previously uploaded firmware related files."""
    self.module.log('(Proxy) Firmware upgrade commencing...')
    body = {'stageFirmware': False, 'skipMelCheck': self.clear_mel_events, 'cfwFile': self.firmware_name}
    if self.nvsram:
        body.update({'nvsramFile': self.nvsram_name})
    try:
        rc, response = self.request('storage-systems/%s/cfw-upgrade' % self.ssid, method='POST', data=body)
    except Exception as error:
        self.module.fail_json(msg='Failed to initiate firmware upgrade. Array [%s]. Error [%s].' % (self.ssid, to_native(error)))
    self.upgrade_in_progress = True
    if self.wait_for_completion:
        self.proxy_wait_for_upgrade()