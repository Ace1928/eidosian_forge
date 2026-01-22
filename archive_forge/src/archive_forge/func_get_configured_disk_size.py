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
def get_configured_disk_size(self, expected_disk_spec):
    if [x for x in expected_disk_spec.keys() if (x.startswith('size_') or x == 'size') and expected_disk_spec[x]]:
        if expected_disk_spec['size']:
            size_regex = re.compile('(\\d+(?:\\.\\d+)?)([tgmkTGMK][bB])')
            disk_size_m = size_regex.match(expected_disk_spec['size'])
            try:
                if disk_size_m:
                    expected = disk_size_m.group(1)
                    unit = disk_size_m.group(2)
                else:
                    raise ValueError
                if re.match('\\d+\\.\\d+', expected):
                    expected = float(expected)
                else:
                    expected = int(expected)
                if not expected or not unit:
                    raise ValueError
            except (TypeError, ValueError, NameError):
                self.module.fail_json(msg='Failed to parse disk size please review value provided using documentation.')
        else:
            param = [x for x in expected_disk_spec.keys() if x.startswith('size_') and expected_disk_spec[x]][0]
            unit = param.split('_')[-1]
            expected = expected_disk_spec[param]
        disk_units = dict(tb=3, gb=2, mb=1, kb=0)
        if unit in disk_units:
            unit = unit.lower()
            return expected * 1024 ** disk_units[unit]
        else:
            self.module.fail_json(msg="%s is not a supported unit for disk size. Supported units are ['%s']." % (unit, "', '".join(disk_units.keys())))
    self.module.fail_json(msg='No size, size_kb, size_mb, size_gb or size_tb defined in disk configuration')