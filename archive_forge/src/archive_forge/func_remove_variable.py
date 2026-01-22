from __future__ import absolute_import, division, print_function
import os
import platform
import pwd
import re
import shlex
import sys
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves import shlex_quote
def remove_variable(self, name):
    self.update_variable(name, None, remove=True)