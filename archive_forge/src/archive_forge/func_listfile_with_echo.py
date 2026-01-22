from __future__ import absolute_import, division, print_function
import os
from tempfile import TemporaryFile, NamedTemporaryFile
from select import select
from abc import ABCMeta, abstractmethod
import tarfile
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.exceptions import (
from ansible.module_utils._text import to_native
def listfile_with_echo(self, path):
    echo_cmd = [self.pod_shell, '-c', 'echo {path}/* {path}/.*'.format(path=path.translate(str.maketrans({' ': '\\ '})))]
    error, out, err = self._run_from_pod(cmd=echo_cmd)
    if error.get('status') != 'Success':
        self.module.fail_json(msg=error.get('message'))
    files = []
    if out:
        output = out[0] + ' '
        files = [os.path.join(path, p[:-1]) for p in output.split(f'{path}/') if p and p[:-1] not in ('.', '..')]
    result = []
    for f in files:
        is_dir, err = self.is_directory_path_from_pod(f)
        if err:
            continue
        if not is_dir:
            result.append(f)
            continue
        result += self.listfile_with_echo(f)
    return result