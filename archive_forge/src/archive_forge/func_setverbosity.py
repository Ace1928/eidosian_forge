from __future__ import absolute_import, division, print_function
import copy
def setverbosity(self, v):
    if not 0 <= v <= 4:
        raise ValueError('verbosity must be an int in the range 0 to 4')
    self._verbosity = v