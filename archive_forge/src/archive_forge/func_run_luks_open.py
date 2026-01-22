from __future__ import (absolute_import, division, print_function)
import os
import re
import stat
from ansible.module_utils.basic import AnsibleModule
def run_luks_open(self, device, keyfile, passphrase, perf_same_cpu_crypt, perf_submit_from_crypt_cpus, perf_no_read_workqueue, perf_no_write_workqueue, persistent, allow_discards, name):
    args = [self._cryptsetup_bin]
    if keyfile:
        args.extend(['--key-file', keyfile])
    if perf_same_cpu_crypt:
        args.extend(['--perf-same_cpu_crypt'])
    if perf_submit_from_crypt_cpus:
        args.extend(['--perf-submit_from_crypt_cpus'])
    if perf_no_read_workqueue:
        args.extend(['--perf-no_read_workqueue'])
    if perf_no_write_workqueue:
        args.extend(['--perf-no_write_workqueue'])
    if persistent:
        args.extend(['--persistent'])
    if allow_discards:
        args.extend(['--allow-discards'])
    args.extend(['open', '--type', 'luks', device, name])
    result = self._run_command(args, data=passphrase)
    if result[RETURN_CODE] != 0:
        raise ValueError('Error while opening LUKS container on %s: %s' % (device, result[STDERR]))