from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_bytes
import re
import os
def subvolume_snapshot(self, snapshot_source, snapshot_destination):
    command = [self.__btrfs, 'subvolume', 'snapshot', to_bytes(snapshot_source), to_bytes(snapshot_destination)]
    result = self.__module.run_command(command, check_rc=True)