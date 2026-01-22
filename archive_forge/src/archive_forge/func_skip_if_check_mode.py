from __future__ import absolute_import, division, print_function
import abc
import os
import stat
import traceback
from ansible.module_utils import six
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.crypto.plugins.module_utils.openssh.utils import (
@staticmethod
def skip_if_check_mode(f):

    def wrapper(self, *args, **kwargs):
        if not self.check_mode:
            f(self, *args, **kwargs)
    return wrapper