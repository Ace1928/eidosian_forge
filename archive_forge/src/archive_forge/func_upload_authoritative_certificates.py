from __future__ import absolute_import, division, print_function
import binascii
import random
import re
from ansible.module_utils import six
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
from ansible.module_utils._text import to_native
from time import sleep
def upload_authoritative_certificates(self, certificate):
    """Install all authoritative certificates."""
    headers, data = create_multipart_formdata([['file', certificate['alias'], certificate['certificate']]])
    try:
        rc, resp = self.request(self.url_path_prefix + 'certificates/server%s&alias=%s' % (self.url_path_suffix, certificate['alias']), method='POST', headers=headers, data=data)
    except Exception as error:
        self.module.fail_json(msg='Failed to upload certificate authority! Array [%s]. Error [%s].' % (self.ssid, error))