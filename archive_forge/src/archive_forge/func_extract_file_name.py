from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import env_fallback
from ansible.module_utils._text import to_native
import os.path
def extract_file_name(self, file_path):
    return os.path.basename(file_path)