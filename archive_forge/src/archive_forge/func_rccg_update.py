from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import \
from ansible.module_utils._text import to_native
def rccg_update(self, modify, modifycv):
    if modify:
        self.log('updating chrcconsistgrp with properties %s', modify)
        cmd = 'chrcconsistgrp'
        cmdopts = {}
        for prop in modify:
            cmdopts[prop] = modify[prop]
        cmdargs = [self.name]
        self.restapi.svc_run_command(cmd, cmdopts, cmdargs)
        self.changed = True
    if modifycv:
        self.log('updating chrcconsistgrp with properties %s', modifycv)
        cmd = 'chrcconsistgrp'
        cmdargs = [self.name]
        for prop in modifycv:
            cmdopts = {}
            cmdopts[prop] = modifycv[prop]
            self.restapi.svc_run_command(cmd, cmdopts, cmdargs)
        self.changed = True
    if not modify and (not modifycv):
        self.log('There is no property to be updated')
        self.changed = False