from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_native
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils.basic import AnsibleModule
from traceback import format_exc
def rcrelationship_probe(self, data):
    props = {}
    propscv = {}
    if data['consistency_group_name'] and self.noconsistgrp:
        props['noconsistgrp'] = self.noconsistgrp
    if self.consistgrp is not None and self.consistgrp != data['consistency_group_name']:
        props['consistgrp'] = self.consistgrp
    if self.master is not None and self.master != data['master_vdisk_name']:
        props['master'] = self.master
    if self.aux is not None and self.aux != data['aux_vdisk_name']:
        props['aux'] = self.aux
    if self.copytype == 'global' and data['copy_type'] == 'metro':
        props['global'] = True
    if (self.copytype == 'metro' or self.copytype is None) and (data['copy_type'] == 'global' and data['cycling_mode'] == 'multi'):
        self.module.fail_json(msg='Changing relationship type from GMCV to metro is not allowed')
    elif (self.copytype == 'metro' or self.copytype is None) and data['copy_type'] == 'global':
        props['metro'] = True
    if self.copytype == 'GMCV' and data['copy_type'] == 'global' and (self.consistgrp is None):
        if data['cycling_mode'] != 'multi':
            propscv['cyclingmode'] = 'multi'
        if self.cyclingperiod is not None and self.cyclingperiod != int(data['cycle_period_seconds']):
            propscv['cycleperiodseconds'] = self.cyclingperiod
    if self.copytype == 'global' and (data['copy_type'] == 'global' and (data['master_change_vdisk_name'] or data['aux_change_vdisk_name'])):
        propscv['cyclingmode'] = 'none'
    if self.copytype == 'GMCV' and data['copy_type'] == 'metro':
        self.module.fail_json(msg='Changing relationship type from metro to GMCV is not allowed')
    if self.copytype != 'metro' and self.copytype != 'global' and (self.copytype != 'GMCV') and (self.copytype is not None):
        self.module.fail_json(msg="Unsupported mirror type: %s. Only 'global', 'metro' and 'GMCV' are supported when modifying" % self.copytype)
    return (props, propscv)