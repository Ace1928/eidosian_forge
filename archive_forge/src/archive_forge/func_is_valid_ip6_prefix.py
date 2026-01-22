from __future__ import absolute_import, division, print_function
import atexit
import time
import re
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils.ansible_release import __version__ as ANSIBLE_VERSION
def is_valid_ip6_prefix(ip6_prefix):
    """Validates given string as IPv6 prefix.

    Args:
        ip6_prefix (str): string to validate as IPv6 prefix.

    Returns:
        bool: True if string is valid IPv6 prefix, else False.
    """
    if not ip6_prefix.isdigit():
        return False
    ip6_prefix_int = int(ip6_prefix)
    if ip6_prefix_int < 0 or ip6_prefix_int > 128:
        return False
    return True