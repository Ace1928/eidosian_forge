from __future__ import absolute_import, division, print_function
import errno
import fnmatch
import grp
import os
import pwd
import re
import stat
import time
from ansible.module_utils.common.text.converters import to_text, to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
def handle_walk_errors(e):
    if e.errno in (errno.EPERM, errno.EACCES):
        skipped[e.filename] = to_text(e)
        return
    raise e