from __future__ import (absolute_import, division, print_function)
import collections
import errno
import glob
import json
import os
import re
import sys
import time
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common.text.formatters import bytes_to_human
from ansible.module_utils.facts.hardware.base import Hardware, HardwareCollector
from ansible.module_utils.facts.utils import get_file_content, get_file_lines, get_mount_size
from ansible.module_utils.six import iteritems
from ansible.module_utils.facts import timeout
def get_lvm_facts(self):
    """ Get LVM Facts if running as root and lvm utils are available """
    lvm_facts = {'lvm': 'N/A'}
    if os.getuid() == 0 and self.module.get_bin_path('vgs'):
        lvm_util_options = '--noheadings --nosuffix --units g --separator ,'
        vgs_path = self.module.get_bin_path('vgs')
        vgs = {}
        if vgs_path:
            rc, vg_lines, err = self.module.run_command('%s %s' % (vgs_path, lvm_util_options))
            for vg_line in vg_lines.splitlines():
                items = vg_line.strip().split(',')
                vgs[items[0]] = {'size_g': items[-2], 'free_g': items[-1], 'num_lvs': items[2], 'num_pvs': items[1]}
        lvs_path = self.module.get_bin_path('lvs')
        lvs = {}
        if lvs_path:
            rc, lv_lines, err = self.module.run_command('%s %s' % (lvs_path, lvm_util_options))
            for lv_line in lv_lines.splitlines():
                items = lv_line.strip().split(',')
                lvs[items[0]] = {'size_g': items[3], 'vg': items[1]}
        pvs_path = self.module.get_bin_path('pvs')
        pvs = {}
        if pvs_path:
            rc, pv_lines, err = self.module.run_command('%s %s' % (pvs_path, lvm_util_options))
            for pv_line in pv_lines.splitlines():
                items = pv_line.strip().split(',')
                pvs[self._find_mapper_device_name(items[0])] = {'size_g': items[4], 'free_g': items[5], 'vg': items[1]}
        lvm_facts['lvm'] = {'lvs': lvs, 'vgs': vgs, 'pvs': pvs}
    return lvm_facts