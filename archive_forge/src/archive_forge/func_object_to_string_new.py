from __future__ import absolute_import, division, print_function
import json
import re
import sys
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.six import binary_type, text_type
from ansible.module_utils._text import to_native
@classmethod
def object_to_string_new(cls, obj):
    output = []
    flds = obj.__dict__
    for k, v in ((k.replace('_', '', 1), v) for k, v in flds.items() if v):
        if isinstance(v, bool):
            output.append('"%s":%s' % (k, v))
        elif isinstance(v, (binary_type, text_type)):
            v = to_native(v, errors='surrogate_or_strict')
            output.append('"%s":"%s"' % (k, v))
        elif isinstance(v, int):
            output.append('"%s":"%s"' % (k, v))
    return ','.join(output)