from __future__ import absolute_import, division, print_function
import copy
import os
import re
import datetime
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.connection import exec_command
from ansible.module_utils.six import iteritems
from ansible.module_utils.parsing.convert_bool import (
from collections import defaultdict
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from .constants import (
def mark_managed_by(ansible_version, params):
    metadata = []
    result = copy.deepcopy(params)
    found1 = False
    found2 = False
    mark1 = dict(name=MANAGED_BY_ANNOTATION_VERSION, value=ansible_version, persist='true')
    mark2 = dict(name=MANAGED_BY_ANNOTATION_MODIFIED, value=str(datetime.datetime.utcnow()), persist='true')
    if 'metadata' not in result:
        result['metadata'] = [mark1, mark2]
        return result
    for x in params['metadata']:
        if x['name'] == MANAGED_BY_ANNOTATION_VERSION:
            found1 = True
            metadata.append(mark1)
        if x['name'] == MANAGED_BY_ANNOTATION_MODIFIED:
            found2 = True
            metadata.append(mark1)
        else:
            metadata.append(x)
    if not found1:
        metadata.append(mark1)
    if not found2:
        metadata.append(mark2)
    result['metadata'] = metadata
    return result