from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def update_images(self):
    if self.uuid == '*':
        cmd = '{0} update'.format(self.cmd)
    else:
        cmd = '{0} update {1}'.format(self.cmd, self.uuid)
    rc, stdout, stderr = self.module.run_command(cmd)
    if rc != 0:
        self.module.fail_json(msg='Failed to update images: {0}'.format(self.errmsg(stderr)))
    self.changed = True