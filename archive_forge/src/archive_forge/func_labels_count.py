import re
from fqdn._compat import cached_property
@property
def labels_count(self):
    has_terminal_dot = self._fqdn[-1] == '.'
    count = self._fqdn.count('.') + (0 if has_terminal_dot else 1)
    return count