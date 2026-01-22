from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_text
import re
def extra_options_validation(self):
    """ Additional validation of options set passed to module that cannot be implemented in module's argspecs. """
    if self.type not in ('bridge-slave', 'team-slave', 'bond-slave'):
        if self.master is None and self.slave_type is not None:
            self.module.fail_json(msg="'master' option is required when 'slave_type' is specified.")