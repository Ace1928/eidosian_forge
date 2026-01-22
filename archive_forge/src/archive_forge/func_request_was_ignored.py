from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.facts.system.chroot import is_chroot
from ansible.module_utils.service import sysv_exists, sysv_is_enabled, fail_if_missing
from ansible.module_utils.common.text.converters import to_native
def request_was_ignored(out):
    return '=' not in out and ('ignoring request' in out or 'ignoring command' in out)