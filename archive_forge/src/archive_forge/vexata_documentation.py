from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.basic import env_fallback
Convert a '<integer>[MGT]' string to MiB, return -1 on error.