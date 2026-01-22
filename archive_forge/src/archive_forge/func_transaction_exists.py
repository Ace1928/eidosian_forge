from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.respawn import has_respawned, respawn_module
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.yumdnf import YumDnf, yumdnf_argument_spec
import errno
import os
import re
import sys
import tempfile
from contextlib import contextmanager
from ansible.module_utils.urls import fetch_file
def transaction_exists(self, pkglist):
    """
        checks the package list to see if any packages are
        involved in an incomplete transaction
        """
    conflicts = []
    if not transaction_helpers:
        return conflicts
    pkglist_nvreas = (splitFilename(pkg) for pkg in pkglist)
    unfinished_transactions = find_unfinished_transactions()
    for trans in unfinished_transactions:
        steps = find_ts_remaining(trans)
        for step in steps:
            action, step_spec = step
            n, v, r, e, a = splitFilename(step_spec)
            for pkg in pkglist_nvreas:
                label = '%s-%s' % (n, a)
                if n == pkg[0] and a == pkg[4]:
                    if label not in conflicts:
                        conflicts.append('%s-%s' % (n, a))
                    break
    return conflicts