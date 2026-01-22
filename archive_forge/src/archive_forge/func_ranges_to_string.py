from __future__ import absolute_import
import cython
from .Transitions import TransitionMap
def ranges_to_string(self, range_list):
    return ','.join(map(self.range_to_string, range_list))