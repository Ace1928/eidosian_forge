from __future__ import absolute_import, division, print_function
import platform
import re
from ansible.module_utils.six import iteritems
from ansible.module_utils.six.moves import zip
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.ios import (
def parse_filesystems_info(self, data):
    facts = dict()
    fs = ''
    for line in data.split('\n'):
        match = re.match('^Directory of (\\S+)/', line)
        if match:
            fs = match.group(1)
            facts[fs] = dict()
            continue
        match = re.match('^(\\d+) bytes total \\((\\d+) bytes free\\)', line)
        if match:
            facts[fs]['spacetotal_kb'] = int(match.group(1)) / 1024
            facts[fs]['spacefree_kb'] = int(match.group(2)) / 1024
    return facts