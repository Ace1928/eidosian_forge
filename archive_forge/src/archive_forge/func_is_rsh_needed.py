from __future__ import absolute_import, division, print_function
import os
import errno
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils.six.moves import shlex_quote
def is_rsh_needed(source, dest):
    if source.startswith('rsync://') or dest.startswith('rsync://'):
        return False
    if ':' in source or ':' in dest:
        return True
    return False