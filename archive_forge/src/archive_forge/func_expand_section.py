from __future__ import absolute_import, division, print_function
import hashlib
import re
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils.six.moves import zip
def expand_section(self, configobj, S=None):
    if S is None:
        S = list()
    S.append(configobj)
    for child in configobj.child_objs:
        if child in S:
            continue
        self.expand_section(child, S)
    return S