from __future__ import absolute_import, division, print_function
import atexit
import time
import re
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils.ansible_release import __version__ as ANSIBLE_VERSION
def is_valid_ip_netmask(ip_netmask):
    """Validates given string as IPv4 netmask.

    Args:
        ip_netmask (str): string to validate as IPv4 netmask.

    Returns:
        bool: True if string is valid IPv4 netmask, else False.
    """
    ip_netmask_split = ip_netmask.split('.')
    if len(ip_netmask_split) != 4:
        return False
    valid_octet_values = ['0', '128', '192', '224', '240', '248', '252', '254', '255']
    for ip_netmask_octet in ip_netmask_split:
        if ip_netmask_octet not in valid_octet_values:
            return False
    if ip_netmask_split[0] != '255' and (ip_netmask_split[1] != '0' or ip_netmask_split[2] != '0' or ip_netmask_split[3] != '0'):
        return False
    elif ip_netmask_split[1] != '255' and (ip_netmask_split[2] != '0' or ip_netmask_split[3] != '0'):
        return False
    elif ip_netmask_split[2] != '255' and ip_netmask_split[3] != '0':
        return False
    return True