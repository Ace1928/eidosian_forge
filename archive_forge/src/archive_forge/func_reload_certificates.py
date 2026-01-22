from __future__ import absolute_import, division, print_function
import binascii
import os
import re
from time import sleep
from datetime import datetime
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule, create_multipart_formdata
from ansible.module_utils._text import to_native
def reload_certificates(self):
    """Reload certificates on both controllers."""
    rc, resp = self.request(self.url_path_prefix + 'certificates/reload?reloadBoth=true', method='POST', ignore_errors=True)
    if rc == 404:
        rc, resp = self.request(self.url_path_prefix + 'sslconfig/reload?reloadBoth=true', method='POST', ignore_errors=True)
    if rc > 202:
        self.module.fail_json(msg='Failed to initiate certificate reload on both controllers! Array [%s].' % self.ssid)
    for retry in range(int(self.RELOAD_TIMEOUT_SEC / 3)):
        rc, current_certificates = self.request(self.url_path_prefix + 'certificates/remote-server', ignore_errors=True)
        if rc == 404:
            rc, current_certificates = self.request(self.url_path_prefix + 'sslconfig/ca?useTruststore=true', ignore_errors=True)
        if rc < 300:
            break
        sleep(3)
    else:
        self.module.fail_json(msg='Failed to retrieve server certificates. Array [%s].' % self.ssid)