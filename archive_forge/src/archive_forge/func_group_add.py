from __future__ import absolute_import, division, print_function
import grp
import os
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.sys_info import get_platform_subclass
def group_add(self, **kwargs):
    cmd = [self.module.get_bin_path('addgroup', True)]
    if self.gid is not None:
        cmd.extend(['-g', str(self.gid)])
    if self.system:
        cmd.append('-S')
    cmd.append(self.name)
    return self.execute_command(cmd)