from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_bytes
import re
import os
def filesystem_show(self):
    command = '%s filesystem show -d' % self.__btrfs
    result = self.__module.run_command(command, check_rc=True)
    stdout = [x.strip() for x in result[1].splitlines()]
    filesystems = []
    current = None
    for line in stdout:
        if line.startswith('Label'):
            current = self.__parse_filesystem(line)
            filesystems.append(current)
        elif line.startswith('devid'):
            current['devices'].append(self.__parse_filesystem_device(line))
    return filesystems