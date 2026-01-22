from __future__ import absolute_import, division, print_function
import json
import re
import sys
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.six import binary_type, text_type
from ansible.module_utils._text import to_native
def get_ns_hardware(client):
    from nssrc.com.citrix.netscaler.nitro.resource.config.ns.nshardware import nshardware
    result = nshardware.get(client)
    return result