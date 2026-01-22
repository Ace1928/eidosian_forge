from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import re
def value_present(self):
    if self.contains():
        return
    if self.module.check_mode:
        self.changed = True
        return
    setstring = '%s+=%s%s' % (self.name, self.delim, self.value)
    rc, out, err = self.run_sysrc(setstring)
    if out.find('%s:' % self.name) == 0:
        values = out.split(' -> ')[1].strip().split(self.delim)
        if self.value in values:
            self.changed = True