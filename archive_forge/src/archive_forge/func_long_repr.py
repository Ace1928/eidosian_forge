from __future__ import unicode_literals
from ._utils import get_hash, get_hash_int
from builtins import object
from collections import namedtuple
def long_repr(self, include_hash=True):
    formatted_props = ['{!r}'.format(arg) for arg in self.args]
    formatted_props += ['{}={!r}'.format(key, self.kwargs[key]) for key in sorted(self.kwargs)]
    out = '{}({})'.format(self.name, ', '.join(formatted_props))
    if include_hash:
        out += ' <{}>'.format(self.short_hash)
    return out