from __future__ import absolute_import, division, print_function
import os
import platform
import re
import tempfile
import time
from ansible.module_utils.basic import AnsibleModule
def state_detached(self):
    if not self.exists():
        self.module.fail_json(msg='zone does not exist')
    if self.is_configured():
        self.msg.append('zone already detached')
    else:
        self.stop()
        self.detach()