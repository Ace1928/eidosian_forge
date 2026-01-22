from __future__ import absolute_import, division, print_function
import json
import re
from copy import deepcopy
from ansible.module_utils._text import to_text
from ansible.module_utils.common._collections_compat import Mapping
from ansible.module_utils.connection import Connection, ConnectionError
from ansible.module_utils.six import PY2, PY3
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
def feature_enable(self):
    """Add 'feature <foo>' to _proposed if ref includes a 'feature' key."""
    ref = self._ref
    feature = ref['_template'].get('feature')
    if feature:
        show_cmd = "show run | incl 'feature {0}'".format(feature)
        output = self.execute_show_command(show_cmd, 'text')
        if not output or 'CLI command error' in output:
            msg = "** 'feature {0}' is not enabled. Module will auto-enable feature {0} ** ".format(feature)
            self._module.warn(msg)
            ref['_proposed'].append('feature {0}'.format(feature))
            ref['_cli_is_feature_disabled'] = ref['_proposed']