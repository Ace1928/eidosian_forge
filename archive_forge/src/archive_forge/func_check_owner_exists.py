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
def check_owner_exists(module, owner):
    try:
        uid = int(owner)
        try:
            getpwuid(uid).pw_name
        except KeyError:
            module.warn('failed to look up user with uid %s. Create user up to this point in real play' % uid)
    except ValueError:
        try:
            getpwnam(owner).pw_uid
        except KeyError:
            module.warn('failed to look up user %s. Create user up to this point in real play' % owner)