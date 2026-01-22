from __future__ import annotations
import collections
import contextlib
import os
import platform
import pstats
import re
import sys
from . import config
from .util import gc_collect
from ..util import has_compiled_ext
def reset_count(self):
    test_key = _current_test
    if test_key not in self.data:
        return
    per_fn = self.data[test_key]
    if self.platform_key not in per_fn:
        return
    per_platform = per_fn[self.platform_key]
    if 'counts' in per_platform:
        per_platform['counts'][:] = []