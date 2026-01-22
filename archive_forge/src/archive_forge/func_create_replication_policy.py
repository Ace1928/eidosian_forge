from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def create_replication_policy(self):
    if self.module.check_mode:
        self.changed = True
        return
    cmd = 'mkreplicationpolicy'
    cmdopts = {'name': self.name, 'topology': self.topology, 'location1system': self.location1system, 'location1iogrp': self.location1iogrp, 'location2system': self.location2system, 'location2iogrp': self.location2iogrp, 'rpoalert': self.rpoalert}
    self.restapi.svc_run_command(cmd, cmdopts, cmdargs=None)
    self.log('Replication policy (%s) created', self.name)
    self.changed = True