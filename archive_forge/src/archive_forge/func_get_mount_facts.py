from __future__ import (absolute_import, division, print_function)
import re
import time
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.text.formatters import bytes_to_human
from ansible.module_utils.facts.utils import get_file_content, get_mount_size
from ansible.module_utils.facts.hardware.base import Hardware, HardwareCollector
from ansible.module_utils.facts import timeout
from ansible.module_utils.six.moves import reduce
@timeout.timeout()
def get_mount_facts(self):
    mount_facts = {}
    mount_facts['mounts'] = []
    fstab = get_file_content('/etc/mnttab')
    if fstab:
        for line in fstab.splitlines():
            fields = line.split('\t')
            mount_statvfs_info = get_mount_size(fields[1])
            mount_info = {'mount': fields[1], 'device': fields[0], 'fstype': fields[2], 'options': fields[3], 'time': fields[4]}
            mount_info.update(mount_statvfs_info)
            mount_facts['mounts'].append(mount_info)
    return mount_facts