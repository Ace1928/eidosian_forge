from __future__ import absolute_import, division, print_function
import os
from tempfile import TemporaryFile, NamedTemporaryFile
from select import select
from abc import ABCMeta, abstractmethod
import tarfile
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.exceptions import (
from ansible.module_utils._text import to_native
def listfiles_with_find(self, path):
    find_cmd = ['find', path, '-type', 'f']
    error, files, err = self._run_from_pod(cmd=find_cmd)
    if error.get('status') != 'Success':
        self.module.fail_json(msg=error.get('message'))
    return files