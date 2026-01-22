from __future__ import absolute_import, division, print_function
import os
import multiprocessing
import threading
from time import sleep
from ansible.module_utils import six
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule, create_multipart_formdata, request
from ansible.module_utils._text import to_native
def check_system_health(self):
    """Ensure E-Series storage system is healthy. Works for both embedded and proxy web services."""
    try:
        rc, response = self.request('storage-systems/%s/health-check' % self.ssid, method='POST')
        return response['successful']
    except Exception as error:
        self.module.fail_json(msg='Health check failed! Array Id [%s]. Error[%s].' % (self.ssid, to_native(error)))