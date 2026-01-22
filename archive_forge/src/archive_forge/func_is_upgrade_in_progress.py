from __future__ import absolute_import, division, print_function
import os
import multiprocessing
import threading
from time import sleep
from ansible.module_utils import six
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule, create_multipart_formdata, request
from ansible.module_utils._text import to_native
def is_upgrade_in_progress(self):
    """Determine whether an upgrade is already in progress."""
    in_progress = False
    if self.is_proxy():
        try:
            rc, status = self.request('storage-systems/%s/cfw-upgrade' % self.ssid)
            in_progress = status['running']
        except Exception as error:
            if 'errorMessage' in to_native(error):
                self.module.warn('Failed to retrieve upgrade status. Array [%s]. Error [%s].' % (self.ssid, error))
                in_progress = False
            else:
                self.module.fail_json(msg='Failed to retrieve upgrade status. Array [%s]. Error [%s].' % (self.ssid, error))
    else:
        in_progress = False
    return in_progress