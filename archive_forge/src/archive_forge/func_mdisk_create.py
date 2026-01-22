from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
def mdisk_create(self):
    if not self.level:
        self.module.fail_json(msg='You must pass in level to the module.')
    if not self.drive:
        self.module.fail_json(msg='You must pass in drive to the module.')
    if not self.mdiskgrp:
        self.module.fail_json(msg='You must pass in mdiskgrp to the module.')
    if self.module.check_mode:
        self.changed = True
        return
    self.log("creating mdisk '%s'", self.name)
    cmd = 'mkarray'
    cmdopts = {}
    if self.level:
        cmdopts['level'] = self.level
    if self.drive:
        cmdopts['drive'] = self.drive
    if self.encrypt:
        cmdopts['encrypt'] = self.encrypt
    cmdopts['name'] = self.name
    cmdargs = [self.mdiskgrp]
    self.log('creating mdisk command=%s opts=%s args=%s', cmd, cmdopts, cmdargs)
    result = self.restapi.svc_run_command(cmd, cmdopts, cmdargs)
    self.log('create mdisk result %s', result)
    if 'message' in result:
        self.changed = True
        self.log('create mdisk result message %s', result['message'])
    else:
        self.module.fail_json(msg='Failed to create mdisk [%s]' % self.name)