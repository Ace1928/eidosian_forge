from __future__ import (absolute_import, division, print_function)
import re
import operator
from functools import reduce
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
from ansible.module_utils._text import to_native
def parse_trigger_fieldpath(self, expression):
    parsed = TRIGGER_CONTAINER.search(expression).groupdict()
    if parsed.get('index'):
        parsed['index'] = int(parsed['index'])
    return parsed