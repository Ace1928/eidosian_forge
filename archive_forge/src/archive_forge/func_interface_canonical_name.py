from __future__ import absolute_import, division, print_function
import json
import re
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.basic import env_fallback
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list, ComplexList
from ansible.module_utils.connection import Connection, ConnectionError
from ansible_collections.community.ciscosmb.plugins.module_utils.ciscosmb_canonical_map import base_interfaces
def interface_canonical_name(interface):
    iftype = interface.rstrip('/\\0123456789. ')
    ifno = interface[len(iftype):].lstrip()
    if iftype in base_interfaces:
        iftype = base_interfaces[iftype]
    interface = iftype + str(ifno)
    return interface