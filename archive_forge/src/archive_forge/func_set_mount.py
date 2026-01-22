from __future__ import absolute_import, division, print_function
import errno
import os
import platform
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.posix.plugins.module_utils.mount import ismount
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils.parsing.convert_bool import boolean
def set_mount(module, args):
    """Set/change a mount point location in fstab."""
    name, backup_lines, changed = _set_mount_save_old(module, args)
    return (name, changed)