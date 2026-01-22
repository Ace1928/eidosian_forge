from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import env_fallback
from ansible.module_utils._text import to_native
import os.path
def have_to_change_to_lowercase(attr):
    return attr in lowercase_change_words