from __future__ import (absolute_import, division, print_function)
from re import compile as re_compile, IGNORECASE
from ansible.errors import AnsibleError
from ansible.parsing.splitter import parse_kv
from ansible.plugins.lookup import LookupBase
def parse_simple_args(self, term):
    """parse the shortcut forms, return True/False"""
    match = SHORTCUT.match(term)
    if not match:
        return False
    dummy, start, end, dummy, stride, dummy, format = match.groups()
    if start is not None:
        try:
            start = int(start, 0)
        except ValueError:
            raise AnsibleError("can't parse start=%s as integer" % start)
    if end is not None:
        try:
            end = int(end, 0)
        except ValueError:
            raise AnsibleError("can't parse end=%s as integer" % end)
    if stride is not None:
        try:
            stride = int(stride, 0)
        except ValueError:
            raise AnsibleError("can't parse stride=%s as integer" % stride)
    if start is not None:
        self.start = start
    if end is not None:
        self.end = end
    if stride is not None:
        self.stride = stride
    if format is not None:
        self.format = format
    return True