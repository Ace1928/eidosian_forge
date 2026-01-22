from __future__ import absolute_import, division, print_function
import re
import traceback
import warnings
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
def parse_flat_interface(entry, non_numeric='hw_eth_ilo'):
    try:
        infoname = 'hw_eth' + str(int(entry['Port']) - 1)
    except Exception:
        infoname = non_numeric
    info = {'macaddress': entry['MAC'].replace('-', ':'), 'macaddress_dash': entry['MAC']}
    return (infoname, info)