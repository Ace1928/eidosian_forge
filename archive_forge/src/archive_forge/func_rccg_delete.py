from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import \
from ansible.module_utils._text import to_native
def rccg_delete(self):
    rccg_data = self.get_existing_rccg()
    if not rccg_data:
        self.module.exit_json(msg="rc consistgrp '%s' did not exist" % self.name, changed=False)
    if self.module.check_mode:
        self.changed = True
        return
    self.log("deleting rc consistgrp '%s'", self.name)
    cmd = 'rmrcconsistgrp'
    cmdopts = {'force': True} if self.force else None
    cmdargs = [self.name]
    self.restapi.svc_run_command(cmd, cmdopts, cmdargs)
    msg = "rc consistgrp '%s' is deleted" % self.name
    self.log(msg)
    self.module.exit_json(msg=msg, changed=True)