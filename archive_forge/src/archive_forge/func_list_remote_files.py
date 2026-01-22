from __future__ import absolute_import, division, print_function
import os
from tempfile import TemporaryFile, NamedTemporaryFile
from select import select
from abc import ABCMeta, abstractmethod
import tarfile
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.exceptions import (
from ansible.module_utils._text import to_native
def list_remote_files(self):
    """
        This method will check if the remote path is a dir or file
        if it is a directory the file list will be updated accordingly
        """
    is_dir, error = self.is_directory_path_from_pod(self.remote_path)
    if error:
        self.module.fail_json(msg=error)
    if not is_dir:
        return [self.remote_path]
    else:
        executables = dict(find=self.listfiles_with_find, echo=self.listfile_with_echo)
        for item in executables:
            error, out, err = self._run_from_pod(item)
            if error.get('status') == 'Success':
                return executables.get(item)(self.remote_path)