import datetime
import locale
import re
import subprocess
import sys
import os
from collections import namedtuple
def get_os(run_lambda):
    from platform import machine
    platform = get_platform()
    if platform == 'win32' or platform == 'cygwin':
        return get_windows_version(run_lambda)
    if platform == 'darwin':
        version = get_mac_version(run_lambda)
        if version is None:
            return None
        return 'macOS {} ({})'.format(version, machine())
    if platform == 'linux':
        desc = get_lsb_version(run_lambda)
        if desc is not None:
            return '{} ({})'.format(desc, machine())
        desc = check_release_file(run_lambda)
        if desc is not None:
            return '{} ({})'.format(desc, machine())
        return '{} ({})'.format(platform, machine())
    return platform