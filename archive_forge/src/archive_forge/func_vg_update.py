from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import \
from ansible.module_utils._text import to_native
def vg_update(self, modify):
    if self.module.check_mode:
        self.changed = True
        return
    self.log("updating volume group '%s' ", self.name)
    cmdargs = [self.name]
    try:
        del modify['snapshotpolicysuspended']
    except KeyError:
        self.log('snapshotpolicysuspended modification not reqiured!!')
    else:
        cmd = 'chvolumegroupsnapshotpolicy'
        cmdopts = {'snapshotpolicysuspended': self.snapshotpolicysuspended}
        self.restapi.svc_run_command(cmd, cmdopts, cmdargs)
    cmd = 'chvolumegroup'
    unmaps = ('noownershipgroup', 'nosafeguardpolicy', 'nosnapshotpolicy', 'noreplicationpolicy')
    for field in unmaps:
        cmdopts = {}
        if field == 'nosafeguardpolicy' and field in modify:
            cmdopts['nosafeguardedpolicy'] = modify.pop('nosafeguardpolicy')
            self.restapi.svc_run_command(cmd, cmdopts, cmdargs)
        elif field in modify:
            cmdopts[field] = modify.pop(field)
            self.restapi.svc_run_command(cmd, cmdopts, cmdargs)
    if modify:
        cmdopts = modify
        self.restapi.svc_run_command(cmd, cmdopts, cmdargs)
    self.changed = True