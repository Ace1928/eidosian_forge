from __future__ import absolute_import, division, print_function
import os
from tempfile import TemporaryFile, NamedTemporaryFile
from select import select
from abc import ABCMeta, abstractmethod
import tarfile
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.exceptions import (
from ansible.module_utils._text import to_native
def is_directory_path_from_pod(self, file_path, failed_if_not_exists=True):
    error, out, err = self._run_from_pod(cmd=['test', '-e', file_path])
    if error.get('status') != 'Success':
        if failed_if_not_exists:
            return (None, '%s does not exist in remote pod filesystem' % file_path)
        return (False, None)
    error, out, err = self._run_from_pod(cmd=['test', '-d', file_path])
    return (error.get('status') == 'Success', None)