from __future__ import (absolute_import, division, print_function)
import json
import re
from datetime import timedelta
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.common.collections import is_sequence
from ansible.module_utils.six.moves.urllib.parse import urlparse
def parse_healthcheck(healthcheck):
    """
    Return dictionary of healthcheck parameters and boolean if
    healthcheck defined in image was requested to be disabled.
    """
    if not healthcheck or not healthcheck.get('test'):
        return (None, None)
    result = normalize_healthcheck(healthcheck, normalize_test=True)
    if result['test'] == ['NONE']:
        return (None, True)
    return (result, False)