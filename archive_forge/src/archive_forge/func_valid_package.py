from __future__ import absolute_import, division, print_function
import json
import os.path
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems, string_types
@classmethod
def valid_package(cls, package):
    """A valid package is either None or alphanumeric."""
    if package is None:
        return True
    return isinstance(package, string_types) and (not cls.INVALID_PACKAGE_REGEX.search(package))