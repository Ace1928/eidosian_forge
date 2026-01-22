from __future__ import (absolute_import, division, print_function)
import json
import re
from datetime import timedelta
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.common.collections import is_sequence
from ansible.module_utils.six.moves.urllib.parse import urlparse
def get_legacy_docker_container_diffs(self):
    """
        Return differences in the docker_container legacy format.
        """
    result = []
    for entry in self._diff:
        item = dict()
        item[entry['name']] = dict(parameter=entry['parameter'], container=entry['active'])
        result.append(item)
    return result