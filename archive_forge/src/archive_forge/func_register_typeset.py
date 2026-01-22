from itertools import chain
from .coretypes import (Unit, int8, int16, int32, int64, uint8, uint16, uint32,
def register_typeset(self, name, typeset):
    if name in self.registry:
        raise TypeError('TypeSet %s already defined with types %s' % (name, self.registry[name].types))
    self.registry[name] = typeset
    return typeset