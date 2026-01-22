from __future__ import absolute_import, division, print_function
import base64
import hashlib
import re
from copy import deepcopy
from functools import partial
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.ios import (
def parse_sshkey(data, user):
    sshregex = 'username %s(\\n\\s+key-hash .+$)+' % user
    sshcfg = re.search(sshregex, data, re.M)
    key_list = []
    if sshcfg:
        match = re.findall('key-hash (\\S+ \\S+(?: .+)?)$', sshcfg.group(), re.M)
        if match:
            key_list = match
    return key_list