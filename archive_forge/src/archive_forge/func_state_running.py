from __future__ import absolute_import, division, print_function
import os
import platform
import re
import tempfile
import time
from ansible.module_utils.basic import AnsibleModule
def state_running(self):
    self.state_present()
    if self.is_running():
        self.msg.append('zone already running')
    else:
        self.boot()