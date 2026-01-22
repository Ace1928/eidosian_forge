from __future__ import absolute_import, division, print_function
import os
import multiprocessing
import threading
from time import sleep
from ansible.module_utils import six
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule, create_multipart_formdata, request
from ansible.module_utils._text import to_native
def proxy_wait_for_upgrade(self):
    """Wait for SANtricity Web Services Proxy to report upgrade complete"""
    self.module.log('(Proxy) Waiting for upgrade to complete...')
    status = {}
    while True:
        try:
            rc, status = self.request('storage-systems/%s/cfw-upgrade' % self.ssid, log_request=False, ignore_errors=True)
        except Exception as error:
            self.module.fail_json(msg='Failed to retrieve firmware upgrade status! Array [%s]. Error[%s].' % (self.ssid, to_native(error)))
        if 'errorMessage' in status:
            self.module.warn('Proxy reported an error. Checking whether upgrade completed. Array [%s]. Error [%s].' % (self.ssid, status['errorMessage']))
            self.wait_for_web_services()
            break
        if not status['running']:
            if status['activationCompletionTime']:
                self.upgrade_in_progress = False
                break
            else:
                self.module.fail_json(msg='Failed to complete upgrade. Array [%s].' % self.ssid)
        sleep(5)