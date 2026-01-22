from __future__ import absolute_import, division, print_function
import json
import re
from ansible.module_utils._text import to_native, to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.netconf import (
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.junos import (
def zeroize(module):
    return exec_rpc(module, tostring(Element('request-system-zeroize')), ignore_warning=False)