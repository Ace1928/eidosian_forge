from __future__ import absolute_import, division, print_function
import os
import platform
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
def split_entry(entry):
    """ splits entry and ensures normalized return"""
    a = entry.split(':')
    d = None
    if entry.lower().startswith('d'):
        d = True
        a.pop(0)
    if len(a) == 2:
        a.append(None)
    t, e, p = a
    t = t.lower()
    if t.startswith('u'):
        t = 'user'
    elif t.startswith('g'):
        t = 'group'
    elif t.startswith('m'):
        t = 'mask'
    elif t.startswith('o'):
        t = 'other'
    else:
        t = None
    return [d, t, e, p]