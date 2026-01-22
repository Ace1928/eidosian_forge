from __future__ import (absolute_import, division, print_function)
from collections.abc import MutableMapping
from ansible.utils.vars import merge_hash
def update_custom_stats(self, which, what, host=None):
    """ allow aggregation of a custom stat"""
    if host is None:
        host = '_run'
    if host not in self.custom or which not in self.custom[host]:
        return self.set_custom_stats(which, what, host)
    if not isinstance(what, type(self.custom[host][which])):
        return None
    if isinstance(what, MutableMapping):
        self.custom[host][which] = merge_hash(self.custom[host][which], what)
    else:
        self.custom[host][which] += what