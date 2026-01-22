from __future__ import absolute_import, division, print_function
import os
import traceback
import xml.etree.ElementTree as ET
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
def has_config_changed(self):
    if self.config is None:
        return False
    config_file = self.get_config()
    machine_file = self.get_current_config()
    self.result['diff']['after'] = config_file
    self.result['diff']['before'] = machine_file
    if machine_file != config_file:
        return True
    return False