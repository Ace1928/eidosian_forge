from __future__ import absolute_import, division, print_function
import shlex
import time
import traceback
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils.common import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible.module_utils.basic import human_to_bytes
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_text, to_native
def get_nanoseconds_from_raw_option(name, value):
    if value is None:
        return None
    elif isinstance(value, int):
        return value
    elif isinstance(value, string_types):
        try:
            return int(value)
        except ValueError:
            return convert_duration_to_nanosecond(value)
    else:
        raise ValueError('Invalid type for %s %s (%s). Only string or int allowed.' % (name, value, type(value)))