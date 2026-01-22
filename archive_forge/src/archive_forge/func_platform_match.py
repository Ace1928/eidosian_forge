from __future__ import (absolute_import, division, print_function)
from collections import defaultdict
import platform
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts import timeout
@classmethod
def platform_match(cls, platform_info):
    if platform_info.get('system', None) == cls._platform:
        return cls
    return None