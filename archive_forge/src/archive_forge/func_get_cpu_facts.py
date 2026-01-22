from __future__ import (absolute_import, division, print_function)
import struct
import time
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.facts.hardware.base import Hardware, HardwareCollector
from ansible.module_utils.facts.sysctl import get_sysctl
def get_cpu_facts(self):
    cpu_facts = {}
    if 'machdep.cpu.brand_string' in self.sysctl:
        cpu_facts['processor'] = self.sysctl['machdep.cpu.brand_string']
        cpu_facts['processor_cores'] = self.sysctl['machdep.cpu.core_count']
    else:
        system_profile = self.get_system_profile()
        cpu_facts['processor'] = '%s @ %s' % (system_profile['Processor Name'], system_profile['Processor Speed'])
        cpu_facts['processor_cores'] = self.sysctl['hw.physicalcpu']
    cpu_facts['processor_vcpus'] = self.sysctl.get('hw.logicalcpu') or self.sysctl.get('hw.ncpu') or ''
    return cpu_facts