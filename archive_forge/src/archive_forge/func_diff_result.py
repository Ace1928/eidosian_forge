from __future__ import absolute_import, division, print_function
import copy
@property
def diff_result(self):
    return None if not (self.diff and self.has_changed) else {'before': self.initial_value, 'after': self.value}