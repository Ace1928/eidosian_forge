from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
def update_os(self):
    """Updates the os with `swupd update`"""
    if self.module.check_mode:
        self.module.exit_json(changed=self._needs_update())
    if not self._needs_update():
        self.msg = 'There are no updates available'
        return
    cmd = self._get_cmd('update')
    self._run_cmd(cmd)
    if self.rc == 0:
        self.changed = True
        self.msg = 'Update successful'
        return
    self.failed = True
    self.msg = 'Failed to check for updates'