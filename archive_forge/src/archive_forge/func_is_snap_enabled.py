from __future__ import absolute_import, division, print_function
import re
import json
import numbers
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.module_helper import StateModuleHelper
from ansible_collections.community.general.plugins.module_utils.snap import snap_runner
def is_snap_enabled(self, snap_name):
    with self.runner('_list name') as ctx:
        rc, out, err = ctx.run(name=snap_name)
    if rc != 0:
        return None
    result = out.splitlines()[1]
    match = self.__disable_re.match(result)
    if not match:
        self.do_raise(msg="Unable to parse 'snap list {0}' output:\n{1}".format(snap_name, out))
    notes = match.group('notes')
    return 'disabled' not in notes.split(',')