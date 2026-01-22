from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
def mdiskgrp_create(self):
    self.create_validation()
    self.log("creating mdisk group '%s'", self.name)
    cmd = 'mkmdiskgrp'
    cmdopts = {}
    if not self.ext:
        self.module.fail_json(msg='You must pass the ext to the module.')
    if self.noquota or self.safeguarded:
        if not self.parentmdiskgrp:
            self.module.fail_json(msg='Required parameter missing: parentmdiskgrp')
    self.check_partnership()
    if self.module.check_mode:
        self.changed = True
        return
    if self.parentmdiskgrp:
        cmdopts['parentmdiskgrp'] = self.parentmdiskgrp
        if self.size:
            cmdopts['size'] = self.size
        if self.unit:
            cmdopts['unit'] = self.unit
        if self.safeguarded:
            cmdopts['safeguarded'] = self.safeguarded
        if self.noquota:
            cmdopts['noquota'] = self.noquota
    else:
        if self.easytier:
            cmdopts['easytier'] = self.easytier
        if self.encrypt:
            cmdopts['encrypt'] = self.encrypt
        if self.ext:
            cmdopts['ext'] = str(self.ext)
    if self.provisioningpolicy:
        cmdopts['provisioningpolicy'] = self.provisioningpolicy
    if self.datareduction:
        cmdopts['datareduction'] = self.datareduction
    if self.replicationpoollinkuid:
        cmdopts['replicationpoollinkuid'] = self.replicationpoollinkuid
    if self.ownershipgroup:
        cmdopts['ownershipgroup'] = self.ownershipgroup
    if self.vdiskprotectionenabled:
        cmdopts['vdiskprotectionenabled'] = self.vdiskprotectionenabled
    if self.etfcmoverallocationmax:
        if '%' not in self.etfcmoverallocationmax and self.etfcmoverallocationmax != 'off':
            cmdopts['etfcmoverallocationmax'] = self.etfcmoverallocationmax + '%'
        else:
            cmdopts['etfcmoverallocationmax'] = self.etfcmoverallocationmax
    if self.warning:
        cmdopts['warning'] = str(self.warning) + '%'
    cmdopts['name'] = self.name
    self.log('creating mdisk group command %s opts %s', cmd, cmdopts)
    result = self.restapi.svc_run_command(cmd, cmdopts, cmdargs=None)
    self.log('creating mdisk group result %s', result)
    if self.replication_partner_clusterid:
        self.set_bit_mask()
    if 'message' in result:
        self.log('creating mdisk group command result message %s', result['message'])
    else:
        self.module.fail_json(msg='Failed to create mdisk group [%s]' % self.name)