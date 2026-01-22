from __future__ import absolute_import, division, print_function
import json
import os.path
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems, string_types
@current_package.setter
def current_package(self, package):
    if not self.valid_package(package):
        self._current_package = None
        self.failed = True
        self.message = 'Invalid package: {0}.'.format(package)
        raise HomebrewException(self.message)
    else:
        self._current_package = package
        return package