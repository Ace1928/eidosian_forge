from __future__ import absolute_import, division, print_function
import errno
import os
import shutil
import sys
import time
from pwd import getpwnam, getpwuid
from grp import getgrnam, getgrgid
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_bytes, to_native
def check_group_exists(module, group):
    try:
        gid = int(group)
        try:
            getgrgid(gid).gr_name
        except KeyError:
            module.warn('failed to look up group with gid %s. Create group up to this point in real play' % gid)
    except ValueError:
        try:
            getgrnam(group).gr_gid
        except KeyError:
            module.warn('failed to look up group %s. Create group up to this point in real play' % group)