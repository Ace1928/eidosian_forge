from __future__ import absolute_import, division, print_function
import os
import platform
import re
import tempfile
import time
from ansible.module_utils.basic import AnsibleModule
def configure_password(self):
    shadow = '%s/root/etc/shadow' % self.path
    if self.root_password:
        f = open(shadow, 'r')
        lines = f.readlines()
        f.close()
        for i in range(0, len(lines)):
            fields = lines[i].split(':')
            if fields[0] == 'root':
                fields[1] = self.root_password
                lines[i] = ':'.join(fields)
        f = open(shadow, 'w')
        for line in lines:
            f.write(line)
        f.close()