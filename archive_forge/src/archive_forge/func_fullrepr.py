from typing import Optional, Tuple, ClassVar, Sequence
from .utils import Serialize
@property
def fullrepr(self):
    return '%s(%r, %r)' % (type(self).__name__, self.name, self.filter_out)