from __future__ import absolute_import, division, print_function
import os
import multiprocessing
import threading
from time import sleep
from ansible.module_utils import six
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule, create_multipart_formdata, request
from ansible.module_utils._text import to_native
def embedded_check_compatibility(self):
    """Verify files are compatible with E-Series storage system."""
    if self.nvsram:
        self.embedded_check_nvsram_compatibility()
    if self.firmware:
        self.embedded_check_bundle_compatibility()