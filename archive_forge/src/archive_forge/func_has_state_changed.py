from __future__ import absolute_import, division, print_function
import os
import traceback
import xml.etree.ElementTree as ET
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
def has_state_changed(self, status):
    if self.enabled is None:
        return False
    return self.enabled is False and status != 'disabled' or (self.enabled is True and status == 'disabled')