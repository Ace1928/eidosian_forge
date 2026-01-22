from __future__ import (absolute_import, division, print_function)
from re import compile as re_compile, IGNORECASE
from ansible.errors import AnsibleError
from ansible.parsing.splitter import parse_kv
from ansible.plugins.lookup import LookupBase
def generate_sequence(self):
    if self.stride >= 0:
        adjust = 1
    else:
        adjust = -1
    numbers = range(self.start, self.end + adjust, self.stride)
    for i in numbers:
        try:
            formatted = self.format % i
            yield formatted
        except (ValueError, TypeError):
            raise AnsibleError('problem formatting %r with %r' % (i, self.format))