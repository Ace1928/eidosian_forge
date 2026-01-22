from __future__ import absolute_import, division, print_function
import re
import json
import numbers
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.module_helper import StateModuleHelper
from ansible_collections.community.general.plugins.module_utils.snap import snap_runner
def retrieve_option_map(self, snap_name):
    with self.runner('get name') as ctx:
        rc, out, err = ctx.run(name=snap_name)
    if rc != 0:
        return {}
    result = out.splitlines()
    if 'has no configuration' in result[0]:
        return {}
    try:
        option_map = self.convert_json_to_map(out)
        return option_map
    except Exception as e:
        self.do_raise(msg="Parsing option map returned by 'snap get {0}' triggers exception '{1}', output:\n'{2}'".format(snap_name, str(e), out))