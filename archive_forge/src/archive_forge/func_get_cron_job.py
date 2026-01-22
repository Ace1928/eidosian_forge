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
def get_cron_job(self, minute, hour, day, month, weekday, job, special, disabled):
    job = job.strip('\r\n')
    if disabled:
        disable_prefix = '#'
    else:
        disable_prefix = ''
    if special:
        if self.cron_file:
            return '%s@%s %s %s' % (disable_prefix, special, self.user, job)
        else:
            return '%s@%s %s' % (disable_prefix, special, job)
    elif self.cron_file:
        return '%s%s %s %s %s %s %s %s' % (disable_prefix, minute, hour, day, month, weekday, self.user, job)
    else:
        return '%s%s %s %s %s %s %s' % (disable_prefix, minute, hour, day, month, weekday, job)