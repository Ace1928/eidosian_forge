from __future__ import absolute_import, division, print_function
import os
import re
import tempfile
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems, string_types
@current_cask.setter
def current_cask(self, cask):
    if not self.valid_cask(cask):
        self._current_cask = None
        self.failed = True
        self.message = 'Invalid cask: {0}.'.format(cask)
        raise HomebrewCaskException(self.message)
    else:
        self._current_cask = cask
        return cask