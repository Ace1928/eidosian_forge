from __future__ import absolute_import, division, print_function
import os
from tempfile import TemporaryFile, NamedTemporaryFile
from select import select
from abc import ABCMeta, abstractmethod
import tarfile
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.exceptions import (
from ansible.module_utils._text import to_native
@property
def pod_shell(self):
    if self._shellname is None:
        for s in ('/bin/sh', '/bin/bash'):
            error, out, err = self._run_from_pod(s)
            if error.get('status') == 'Success':
                self._shellname = s
                break
    return self._shellname