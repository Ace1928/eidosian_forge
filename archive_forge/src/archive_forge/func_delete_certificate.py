from __future__ import absolute_import, division, print_function
import binascii
import os
import re
from time import sleep
from datetime import datetime
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule, create_multipart_formdata
from ansible.module_utils._text import to_native
def delete_certificate(self, info):
    """Delete existing remote server certificate in the storage array truststore."""
    rc, resp = self.request(self.url_path_prefix + 'certificates/remote-server/%s' % info['alias'], method='DELETE', ignore_errors=True)
    if rc == 404:
        rc, resp = self.request(self.url_path_prefix + 'sslconfig/ca/%s?useTruststore=true' % info['alias'], method='DELETE', ignore_errors=True)
    if rc > 204:
        self.module.fail_json(msg='Failed to delete certificate. Alias [%s]. Array [%s]. Error [%s, %s].' % (info['alias'], self.ssid, rc, resp))