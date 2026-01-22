from __future__ import absolute_import, division, print_function
from copy import deepcopy
import re
import os
import ast
import datetime
import shutil
import tempfile
from ansible.module_utils.basic import json
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.six import PY3
from ansible.module_utils.six.moves import filterfalse
from ansible.module_utils.six.moves.urllib.parse import urlencode, urljoin
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_native, to_text
from ansible.module_utils.connection import Connection
from ansible_collections.cisco.mso.plugins.module_utils.constants import NDO_API_VERSION_PATH_FORMAT
def make_dhcp_label(self, data):
    """Create a DHCP policy from input"""
    if data is None:
        return None
    if type(data) == list:
        dhcps = []
        for dhcp in data:
            if 'dhcp_option_policy' in dhcp:
                dhcp['dhcpOptionLabel'] = dhcp.get('dhcp_option_policy')
                del dhcp['dhcp_option_policy']
            dhcps.append(dhcp)
        return dhcps
    if 'version' in data:
        data['version'] = int(data.get('version'))
    if data and 'dhcp_option_policy' in data:
        dhcp_option_policy = data.get('dhcp_option_policy')
        if dhcp_option_policy is not None and 'version' in dhcp_option_policy:
            dhcp_option_policy['version'] = int(dhcp_option_policy.get('version'))
        data['dhcpOptionLabel'] = dhcp_option_policy
        del data['dhcp_option_policy']
    return data