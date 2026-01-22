from __future__ import absolute_import, division, print_function
import binascii
import random
import re
from ansible.module_utils import six
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
from ansible.module_utils._text import to_native
from time import sleep
def remove_authoritative_certificates(self, alias):
    """Delete all authoritative certificates."""
    try:
        rc, resp = self.request(self.url_path_prefix + 'certificates/server/%s%s' % (alias, self.url_path_suffix), method='DELETE')
    except Exception as error:
        self.module.fail_json(msg='Failed to delete certificate authority! Array [%s]. Error [%s].' % (self.ssid, error))