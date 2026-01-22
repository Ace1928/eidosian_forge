from __future__ import (absolute_import, division, print_function)
import json
import re
from datetime import timedelta
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.common.collections import is_sequence
from ansible.module_utils.six.moves.urllib.parse import urlparse
def get_legacy_docker_diffs(self):
    """
        Return differences in the docker_container legacy format.
        """
    result = [entry['name'] for entry in self._diff]
    return result