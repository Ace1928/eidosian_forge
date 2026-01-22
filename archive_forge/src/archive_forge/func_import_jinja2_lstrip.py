from __future__ import absolute_import, division, print_function
import copy
import traceback
import os
from contextlib import contextmanager
import platform
from ansible.config.manager import ensure_type
from ansible.errors import (
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import string_types, iteritems
from ansible.module_utils._text import to_text, to_bytes, to_native
from ansible.plugins.action import ActionBase
def import_jinja2_lstrip(self, templates):
    if any((tmp['lstrip_blocks'] for tmp in templates)):
        try:
            import jinja2.defaults
        except ImportError:
            raise AnsibleError('Unable to import Jinja2 defaults for determining Jinja2 features.')
        try:
            jinja2.defaults.LSTRIP_BLOCKS
        except AttributeError:
            raise AnsibleError("Option `lstrip_blocks' is only available in Jinja2 versions >=2.7")