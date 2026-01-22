from __future__ import absolute_import, division, print_function
import os
import multiprocessing
import threading
from time import sleep
from ansible.module_utils import six
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule, create_multipart_formdata, request
from ansible.module_utils._text import to_native
def proxy_check_firmware_compatibility(self, retries=10):
    """Verify firmware is compatible with E-Series storage system."""
    check = {}
    try:
        rc, check = self.request('firmware/compatibility-check', method='POST', data={'storageDeviceIds': [self.ssid]})
    except Exception as error:
        if retries:
            sleep(1)
            self.proxy_check_firmware_compatibility(retries - 1)
        else:
            self.module.fail_json(msg='Failed to receive firmware compatibility information. Array [%s]. Error [%s].' % (self.ssid, to_native(error)))
    for count in range(int(self.COMPATIBILITY_CHECK_TIMEOUT_SEC / 5)):
        try:
            rc, response = self.request('firmware/compatibility-check?requestId=%s' % check['requestId'])
        except Exception as error:
            continue
        if not response['checkRunning']:
            for result in response['results'][0]['cfwFiles']:
                if result['filename'] == self.firmware_name:
                    return
            self.module.fail_json(msg='Firmware bundle is not compatible. firmware [%s]. Array [%s].' % (self.firmware_name, self.ssid))
        sleep(5)
    self.module.fail_json(msg='Failed to retrieve firmware status update from proxy. Array [%s].' % self.ssid)