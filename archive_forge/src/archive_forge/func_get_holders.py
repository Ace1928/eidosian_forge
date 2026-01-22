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
def get_holders(self, block_dev_dict, sysdir):
    block_dev_dict['holders'] = []
    if os.path.isdir(sysdir + '/holders'):
        for folder in os.listdir(sysdir + '/holders'):
            if not folder.startswith('dm-'):
                continue
            name = get_file_content(sysdir + '/holders/' + folder + '/dm/name')
            if name:
                block_dev_dict['holders'].append(name)
            else:
                block_dev_dict['holders'].append(folder)