from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def vdisk_create(self, data):
    if not self.remote_pool:
        self.module.fail_json(msg='You must pass in remote_pool to the module.')
    if self.module.check_mode:
        self.changed = True
        return
    self.log("creating vdisk '%s'", self.source_volume)
    size = int(data[0]['capacity'])
    cmd = 'mkvolume'
    cmdopts = {}
    if self.remote_pool:
        cmdopts['pool'] = self.remote_pool
    cmdopts['name'] = self.target_volume
    cmdopts['size'] = size
    cmdopts['unit'] = 'b'
    self.log('creating vdisk command %s opts %s', cmd, cmdopts)
    remote_restapi = self.construct_remote_rest()
    result = remote_restapi.svc_run_command(cmd, cmdopts, cmdargs=None)
    self.log('create vdisk result %s', result)
    if 'message' in result:
        self.changed = True
        self.log('create vdisk result message %s', result['message'])
    else:
        self.module.fail_json(msg='Failed to create volume [%s]' % self.source_volume)