from __future__ import (absolute_import, division, print_function)
import struct
import time
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.facts.hardware.base import Hardware, HardwareCollector
from ansible.module_utils.facts.sysctl import get_sysctl
def get_memory_facts(self):
    memory_facts = {'memtotal_mb': int(self.sysctl['hw.memsize']) // 1024 // 1024, 'memfree_mb': 0}
    total_used = 0
    page_size = 4096
    try:
        vm_stat_command = get_bin_path('vm_stat')
    except ValueError:
        return memory_facts
    rc, out, err = self.module.run_command(vm_stat_command)
    if rc == 0:
        memory_stats = (line.rstrip('.').split(':', 1) for line in out.splitlines())
        memory_stats = dict(((k, v.lstrip()) for k, v in memory_stats))
        for k, v in memory_stats.items():
            try:
                memory_stats[k] = int(v)
            except ValueError:
                pass
        if memory_stats.get('Pages wired down'):
            total_used += memory_stats['Pages wired down'] * page_size
        if memory_stats.get('Pages active'):
            total_used += memory_stats['Pages active'] * page_size
        if memory_stats.get('Pages inactive'):
            total_used += memory_stats['Pages inactive'] * page_size
        memory_facts['memfree_mb'] = memory_facts['memtotal_mb'] - total_used // 1024 // 1024
    return memory_facts