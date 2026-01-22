from __future__ import absolute_import, division, print_function
import os
import multiprocessing
import threading
from time import sleep
from ansible.module_utils import six
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule, create_multipart_formdata, request
from ansible.module_utils._text import to_native
def firmware_event_logger(self):
    """Determine if firmware activation has started."""
    try:
        rc, events = self.request('storage-systems/%s/events' % self.ssid)
        for event in events:
            if int(event['eventNumber']) > int(self.last_known_event):
                self.last_known_event = event['eventNumber']
    except Exception as error:
        self.module.fail_json(msg='Failed to determine last known event. Array Id [%s]. Error[%s].' % (self.ssid, to_native(error)))
    while True:
        try:
            rc, events = self.request('storage-systems/%s/events?lastKnown=%s&wait=1' % (self.ssid, self.last_known_event), log_request=False)
            for event in events:
                if int(event['eventNumber']) > int(self.last_known_event):
                    self.last_known_event = event['eventNumber']
                if event['eventType'] == 'firmwareDownloadEvent':
                    self.module.log('%s' % event['status'])
                    if event['status'] == 'informational' and event['statusMessage']:
                        self.module.log('Controller firmware: %s Array Id [%s].' % (event['statusMessage'], self.ssid))
                    if event['status'] == 'activate_success':
                        self.module.log('Controller firmware activated. Array Id [%s].' % self.ssid)
                        return
        except Exception as error:
            pass