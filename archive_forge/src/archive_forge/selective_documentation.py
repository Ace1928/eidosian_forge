from __future__ import (absolute_import, division, print_function)
import difflib
from ansible import constants as C
from ansible.plugins.callback import CallbackBase
from ansible.module_utils.common.text.converters import to_text
Run when a task is skipped.