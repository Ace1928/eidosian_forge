from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
@property
def at_revision(self):
    """
        There is no point in pulling from a potentially down/slow remote site
        if the desired changeset is already the current changeset.
        """
    if self.revision is None or len(self.revision) < 7:
        return False
    rc, out, err = self._command(['--debug', 'id', '-i', '-R', self.dest])
    if rc != 0:
        self.module.fail_json(msg=err)
    if out.startswith(self.revision):
        return True
    return False