from __future__ import (absolute_import, division, print_function)
import os
import re
from ansible.module_utils.facts.hardware.base import Hardware, HardwareCollector
def get_hw_facts(self, collected_facts=None):
    hw_facts = {}
    collected_facts = collected_facts or {}
    rc, out, err = self.module.run_command('model')
    hw_facts['model'] = out.strip()
    if collected_facts.get('ansible_architecture') == 'ia64':
        separator = ':'
        if collected_facts.get('ansible_distribution_version') == 'B.11.23':
            separator = '='
        rc, out, err = self.module.run_command("/usr/contrib/bin/machinfo |grep -i 'Firmware revision' | grep -v BMC", use_unsafe_shell=True)
        hw_facts['firmware_version'] = out.split(separator)[1].strip()
        rc, out, err = self.module.run_command("/usr/contrib/bin/machinfo |grep -i 'Machine serial number' ", use_unsafe_shell=True)
        if rc == 0 and out:
            hw_facts['product_serial'] = out.split(separator)[1].strip()
    return hw_facts