from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def change_volume_create(self):
    if self.module.check_mode:
        self.changed = True
        return
    if not self.basevolume:
        self.module.fail_json(msg='You must pass in name of the master or auxiliary volume.')
    vdisk_data = self.get_existing_vdisk(self.basevolume)
    if not vdisk_data:
        self.module.fail_json(msg='%s volume does not exist, change volume not created' % self.basevolume)
    cmd = 'mkvdisk'
    cmdopts = {}
    cmdopts['name'] = self.cvname
    cmdopts['mdiskgrp'] = vdisk_data['mdisk_grp_name']
    cmdopts['size'] = vdisk_data['capacity']
    cmdopts['unit'] = 'b'
    cmdopts['rsize'] = '0%'
    cmdopts['autoexpand'] = True
    cmdopts['iogrp'] = vdisk_data['IO_group_name']
    self.log('creating vdisk command %s opts %s', cmd, cmdopts)
    result = self.restapi.svc_run_command(cmd, cmdopts, cmdargs=None)
    if 'message' in result:
        self.changed = True
        self.log('Create vdisk result message %s', result['message'])
    else:
        self.module.fail_json(msg='Failed to create vdisk [%s]' % self.cvname)