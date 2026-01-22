from __future__ import absolute_import, division, print_function
import binascii
import random
import re
from ansible.module_utils import six
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
from ansible.module_utils._text import to_native
from time import sleep
@staticmethod
def sanitize_distinguished_name(dn):
    """Generate a sorted distinguished name string to account for different formats/orders."""
    dn = re.sub(' *= *', '=', dn).lower()
    dn = re.sub(', *(?=[a-zA-Z]+={1})', '---SPLIT_MARK---', dn)
    dn_parts = dn.split('---SPLIT_MARK---')
    dn_parts.sort()
    return ','.join(dn_parts)