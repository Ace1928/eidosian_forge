from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import \
from ansible.module_utils._text import to_native
def vg_create(self):
    self.create_validation()
    if self.module.check_mode:
        self.changed = True
        return
    self.log("creating volume group '%s'", self.name)
    cmd = 'mkvolumegroup'
    cmdopts = {'name': self.name, 'safeguarded': self.safeguarded}
    if self.type:
        optional_params = ('type', 'snapshot', 'pool')
        cmdopts.update(dict(((param, getattr(self, param)) for param in optional_params if getattr(self, param))))
        if self.iogrp:
            cmdopts['iogroup'] = self.iogrp
        self.set_parentuid()
        if self.parentuid:
            cmdopts['fromsourceuid'] = self.parentuid
        else:
            cmdopts['fromsourcegroup'] = self.fromsourcegroup
    if self.ignoreuserfcmaps:
        if self.ignoreuserfcmaps == 'yes':
            cmdopts['ignoreuserfcmaps'] = True
        else:
            cmdopts['ignoreuserfcmaps'] = False
    if self.replicationpolicy:
        cmdopts['replicationpolicy'] = self.replicationpolicy
    if self.ownershipgroup:
        cmdopts['ownershipgroup'] = self.ownershipgroup
    elif self.safeguardpolicyname:
        cmdopts['safeguardedpolicy'] = self.safeguardpolicyname
        if self.policystarttime:
            cmdopts['policystarttime'] = self.policystarttime
    elif self.snapshotpolicy:
        cmdopts['snapshotpolicy'] = self.snapshotpolicy
        if self.policystarttime:
            cmdopts['policystarttime'] = self.policystarttime
    self.log("creating volumegroup '%s'", cmdopts)
    result = self.restapi.svc_run_command(cmd, cmdopts, cmdargs=None)
    self.log('create volume group result %s', result)
    self.changed = True