from __future__ import absolute_import, division, print_function
from os.path import isfile
from os import getuid, unlink
import re
import shutil
import tempfile
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six.moves import configparser
from ansible.module_utils import distro
def update_syspurpose(self, new_syspurpose):
    """
        Try to update current syspurpose with new attributes from new_syspurpose
        """
    syspurpose = {}
    syspurpose_changed = False
    for key, value in new_syspurpose.items():
        if key in self.ALLOWED_ATTRIBUTES:
            if value is not None:
                syspurpose[key] = value
        elif key == 'sync':
            pass
        else:
            raise KeyError('Attribute: %s not in list of allowed attributes: %s' % (key, self.ALLOWED_ATTRIBUTES))
    current_syspurpose = self._read_syspurpose()
    if current_syspurpose != syspurpose:
        syspurpose_changed = True
    current_syspurpose.update(syspurpose)
    for key in list(current_syspurpose):
        if key in self.ALLOWED_ATTRIBUTES and key not in syspurpose:
            del current_syspurpose[key]
    self._write_syspurpose(current_syspurpose)
    return syspurpose_changed