from __future__ import absolute_import, division, print_function
import re
import time
import string
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.network import is_mac
from ansible.module_utils._text import to_text, to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible_collections.community.vmware.plugins.module_utils.vm_device_helper import PyVmomiDeviceHelper
from ansible_collections.community.vmware.plugins.module_utils.vmware_spbm import SPBM
def sanitize_cdrom_params(self):
    cdrom_specs = []
    expected_cdrom_spec = self.params.get('cdrom')
    if expected_cdrom_spec:
        for cdrom_spec in expected_cdrom_spec:
            cdrom_spec['controller_type'] = cdrom_spec.get('controller_type')
            cdrom_spec['state'] = cdrom_spec.get('state')
            if cdrom_spec['state'] == 'present':
                cdrom_spec['type'] = cdrom_spec.get('type')
                if cdrom_spec['type'] == 'iso' and (not cdrom_spec.get('iso_path')):
                    self.module.fail_json(msg='cdrom.iso_path is mandatory when cdrom.type is set to iso.')
            if 'controller_number' not in cdrom_spec or 'unit_number' not in cdrom_spec:
                self.module.fail_json(msg="'cdrom.controller_number' and 'cdrom.unit_number' are required parameters when configure CDROM list.")
            cdrom_ctl_num = int(cdrom_spec.get('controller_number'))
            cdrom_ctl_unit_num = int(cdrom_spec.get('unit_number'))
            if cdrom_spec['controller_type'] == 'ide' and (cdrom_ctl_num not in [0, 1] or cdrom_ctl_unit_num not in [0, 1]):
                self.module.fail_json(msg='Invalid cdrom.controller_number: %s or cdrom.unit_number: %s, valid values are 0 or 1 for IDE controller.' % (cdrom_spec.get('controller_number'), cdrom_spec.get('unit_number')))
            if cdrom_spec['controller_type'] == 'sata' and (cdrom_ctl_num not in range(0, 4) or cdrom_ctl_unit_num not in range(0, 30)):
                self.module.fail_json(msg='Invalid cdrom.controller_number: %s or cdrom.unit_number: %s, valid controller_number value is 0-3, valid unit_number is 0-29 for SATA controller.' % (cdrom_spec.get('controller_number'), cdrom_spec.get('unit_number')))
            cdrom_spec['controller_number'] = cdrom_ctl_num
            cdrom_spec['unit_number'] = cdrom_ctl_unit_num
            ctl_exist = False
            for exist_spec in cdrom_specs:
                if exist_spec.get('ctl_num') == cdrom_spec['controller_number'] and exist_spec.get('ctl_type') == cdrom_spec['controller_type']:
                    for cdrom_same_ctl in exist_spec['cdroms']:
                        if cdrom_same_ctl['unit_number'] == cdrom_spec['unit_number']:
                            self.module.fail_json(msg='Duplicate cdrom.controller_type: %s, cdrom.controller_number: %s,cdrom.unit_number: %s parameters specified.' % (cdrom_spec['controller_type'], cdrom_spec['controller_number'], cdrom_spec['unit_number']))
                    ctl_exist = True
                    exist_spec['cdroms'].append(cdrom_spec)
                    break
            if not ctl_exist:
                cdrom_specs.append({'ctl_num': cdrom_spec['controller_number'], 'ctl_type': cdrom_spec['controller_type'], 'cdroms': [cdrom_spec]})
    return cdrom_specs