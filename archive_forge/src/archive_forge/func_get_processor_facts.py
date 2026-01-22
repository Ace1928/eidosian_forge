from __future__ import (absolute_import, division, print_function)
import re
import time
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.facts.hardware.base import Hardware, HardwareCollector
from ansible.module_utils.facts import timeout
from ansible.module_utils.facts.utils import get_file_content, get_mount_size
from ansible.module_utils.facts.sysctl import get_sysctl
def get_processor_facts(self):
    cpu_facts = {}
    processor = []
    for i in range(int(self.sysctl['hw.ncpuonline'])):
        processor.append(self.sysctl['hw.model'])
    cpu_facts['processor'] = processor
    cpu_facts['processor_count'] = self.sysctl['hw.ncpuonline']
    cpu_facts['processor_cores'] = self.sysctl['hw.ncpuonline']
    return cpu_facts