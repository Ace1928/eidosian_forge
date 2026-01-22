from __future__ import absolute_import, division, print_function
import os
import platform
import pwd
import re
import sys
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_bytes, to_native
from ansible.module_utils.six.moves import shlex_quote
def get_jobnames(self):
    jobnames = []
    for l in self.lines:
        if re.match('%s' % self.ansible, l):
            jobnames.append(re.sub('%s' % self.ansible, '', l))
    return jobnames