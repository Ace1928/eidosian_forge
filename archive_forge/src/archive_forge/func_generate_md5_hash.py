from __future__ import absolute_import, division, print_function
import collections
import os
from copy import deepcopy
from functools import partial
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.iosxr import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.utils.utils import Version
def generate_md5_hash(self, arg):
    """
        Generate MD5 hash with randomly generated salt size of 3.
        :param arg:
        :return passwd:
        """
    cmd = 'openssl passwd -salt `openssl rand -base64 3` -1 '
    return os.popen(cmd + arg).readlines()[0].strip()