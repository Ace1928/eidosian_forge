import sys
import pytest
def stash(self):
    self._data[self.namespace] = self.modules.pop(self.namespace, None)
    for module in list(self.modules.keys()):
        if module.startswith(self.namespace + '.'):
            self._data[module] = self.modules.pop(module)