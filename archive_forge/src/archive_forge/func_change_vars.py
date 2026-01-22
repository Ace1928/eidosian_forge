from __future__ import absolute_import, division, print_function
import copy
def change_vars(self):
    return [v for v in self._data if self.meta(v).change]