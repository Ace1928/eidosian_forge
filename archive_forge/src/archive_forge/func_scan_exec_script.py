from __future__ import (absolute_import, division, print_function)
import base64
import errno
import json
import os
import pkgutil
import random
import re
from ansible.module_utils.compat.version import LooseVersion
from ansible import constants as C
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.compat.importlib import import_module
from ansible.plugins.loader import ps_module_utils_loader
from ansible.utils.collection_loader import resource_from_fqcr
def scan_exec_script(self, name):
    name = to_text(name)
    if name in self.exec_scripts.keys():
        return
    data = pkgutil.get_data('ansible.executor.powershell', to_native(name + '.ps1'))
    if data is None:
        raise AnsibleError("Could not find executor powershell script for '%s'" % name)
    b_data = to_bytes(data)
    if C.DEFAULT_DEBUG:
        exec_script = b_data
    else:
        exec_script = _strip_comments(b_data)
    self.exec_scripts[name] = to_bytes(exec_script)
    self.scan_module(b_data, wrapper=True, powershell=True)