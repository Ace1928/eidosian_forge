import collections
import os
import stat
import struct
import sys
from typing import (
from dulwich.file import GitFile
from dulwich.objects import (
from dulwich.pack import (
def iterblobs(self):
    import warnings
    warnings.warn('Use iterobjects() instead.', PendingDeprecationWarning)
    return self.iterobjects()