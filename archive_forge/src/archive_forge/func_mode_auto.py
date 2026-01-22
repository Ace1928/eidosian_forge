from __future__ import absolute_import, division, print_function
import os
import re
from ansible.module_utils.basic import AnsibleModule
@property
def mode_auto(self):
    return self.module.params.get('state') == AlternativeState.AUTO