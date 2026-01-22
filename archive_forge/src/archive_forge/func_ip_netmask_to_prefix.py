from __future__ import absolute_import, division, print_function
import atexit
import time
import re
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils.ansible_release import __version__ as ANSIBLE_VERSION
def ip_netmask_to_prefix(ip_netmask, skip_check=False):
    """Converts IPv4 netmask to prefix.

    Args:
        ip_netmask (str): IPv4 netmask to convert.
        skip_check (bool): Skip validation of IPv4 netmask
            (default: False). Use if you are sure IPv4 netmask is valid.

    Returns:
        str: IPv4 prefix equivalent to given IPv4 netmask if
        IPv4 netmask is valid, else an empty string.
    """
    if skip_check:
        ip_netmask_valid = True
    else:
        ip_netmask_valid = is_valid_ip_netmask(ip_netmask)
    if ip_netmask_valid:
        return str(sum([bin(int(i)).count('1') for i in ip_netmask.split('.')]))
    else:
        return ''