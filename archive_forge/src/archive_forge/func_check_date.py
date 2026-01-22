from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.api import basic_auth_argument_spec
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible_collections.community.general.plugins.module_utils.gitlab import (
from datetime import datetime
def check_date(self, _date):
    try:
        datetime.strptime(_date, '%Y-%m-%d')
    except ValueError:
        self._module.fail_json(msg="milestone's date '%s' not in correct format." % _date)
    return _date