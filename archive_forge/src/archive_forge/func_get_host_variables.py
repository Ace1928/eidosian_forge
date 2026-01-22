from __future__ import (absolute_import, division, print_function)
import os
import subprocess
from collections.abc import Mapping
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.module_utils.basic import json_dict_bytes_to_unicode
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.plugins.inventory import BaseInventoryPlugin
from ansible.utils.display import Display
def get_host_variables(self, path, host):
    """ Runs <script> --host <hostname>, to determine additional host variables """
    cmd = [path, '--host', host]
    try:
        sp = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as e:
        raise AnsibleError('problem running %s (%s)' % (' '.join(cmd), e))
    out, stderr = sp.communicate()
    if sp.returncode != 0:
        raise AnsibleError('Inventory script (%s) had an execution error: %s' % (path, to_native(stderr)))
    if out.strip() == '':
        return {}
    try:
        return json_dict_bytes_to_unicode(self.loader.load(out, file_name=path))
    except ValueError:
        raise AnsibleError('could not parse post variable response: %s, %s' % (cmd, out))