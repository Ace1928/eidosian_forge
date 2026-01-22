from __future__ import (absolute_import, division, print_function)
import abc
import re
from os.path import basename
from ansible.module_utils import six
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils import deps
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def get_pids_by_pattern(self, pattern, ignore_case):
    flags = 0
    if ignore_case:
        flags |= re.I
    try:
        regex = re.compile(pattern, flags)
    except re.error as e:
        raise PSAdapterError("'%s' is not a valid regular expression: %s" % (pattern, to_native(e)))
    return [p.pid for p in self._process_iter(*self.PATTERN_ATTRS) if self._matches_regex(p, regex)]