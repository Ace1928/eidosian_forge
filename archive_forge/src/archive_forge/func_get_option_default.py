from __future__ import (absolute_import, division, print_function)
import os
import re
import shutil
import tempfile
import types
from ansible.module_utils.six.moves import configparser
def get_option_default(self, key, default=''):
    sect, opt = key.split('.', 1)
    if self.has_section(sect) and self.has_option(sect, opt):
        return self.get(sect, opt)
    else:
        return default