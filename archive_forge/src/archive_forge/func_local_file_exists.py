from __future__ import (absolute_import, division, print_function)
import re
import os
import sys
import time
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, get_nc_config
from ansible.module_utils.connection import ConnectionError
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import validate_ip_v6_address
def local_file_exists(self):
    """Local file whether exists"""
    return os.path.isfile(self.local_file)