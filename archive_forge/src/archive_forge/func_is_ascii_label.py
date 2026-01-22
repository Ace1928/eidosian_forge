from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.common.text.converters import to_text
def is_ascii_label(domain):
    """
    Check whether domain name has only ASCII labels.
    """
    return _ASCII_PRINTABLE_MATCHER.match(domain) is not None