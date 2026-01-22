from __future__ import absolute_import, division, print_function
import os
import re
import tempfile
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems, string_types
@brew_path.setter
def brew_path(self, brew_path):
    if not self.valid_brew_path(brew_path):
        self._brew_path = None
        self.failed = True
        self.message = 'Invalid brew_path: {0}.'.format(brew_path)
        raise HomebrewCaskException(self.message)
    else:
        self._brew_path = brew_path
        return brew_path