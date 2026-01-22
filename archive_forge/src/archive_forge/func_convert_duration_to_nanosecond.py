from __future__ import (absolute_import, division, print_function)
import json
import re
from datetime import timedelta
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.common.collections import is_sequence
from ansible.module_utils.six.moves.urllib.parse import urlparse
def convert_duration_to_nanosecond(time_str):
    """
    Return time duration in nanosecond.
    """
    if not isinstance(time_str, str):
        raise ValueError('Missing unit in duration - %s' % time_str)
    regex = re.compile('^(((?P<hours>\\d+)h)?((?P<minutes>\\d+)m(?!s))?((?P<seconds>\\d+)s)?((?P<milliseconds>\\d+)ms)?((?P<microseconds>\\d+)us)?)$')
    parts = regex.match(time_str)
    if not parts:
        raise ValueError('Invalid time duration - %s' % time_str)
    parts = parts.groupdict()
    time_params = {}
    for name, value in parts.items():
        if value:
            time_params[name] = int(value)
    delta = timedelta(**time_params)
    time_in_nanoseconds = (delta.microseconds + (delta.seconds + delta.days * 24 * 3600) * 10 ** 6) * 10 ** 3
    return time_in_nanoseconds