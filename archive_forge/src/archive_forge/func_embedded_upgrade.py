from __future__ import absolute_import, division, print_function
import os
import multiprocessing
import threading
from time import sleep
from ansible.module_utils import six
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule, create_multipart_formdata, request
from ansible.module_utils._text import to_native
def embedded_upgrade(self):
    """Upload and activate both firmware and NVSRAM."""
    download_thread = threading.Thread(target=self.embedded_firmware_download)
    event_thread = threading.Thread(target=self.firmware_event_logger)
    download_thread.start()
    event_thread.start()
    download_thread.join()
    event_thread.join()