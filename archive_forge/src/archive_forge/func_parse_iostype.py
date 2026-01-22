from __future__ import absolute_import, division, print_function
import platform
import re
from ansible.module_utils.six import iteritems
from ansible.module_utils.six.moves import zip
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.ios import (
def parse_iostype(self, data):
    match = re.search('\\sIOS-XE\\s', data)
    if match:
        return 'IOS-XE'
    else:
        return 'IOS'