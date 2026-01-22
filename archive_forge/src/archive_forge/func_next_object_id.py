from __future__ import annotations
import calendar
import codecs
import collections
import mmap
import os
import re
import time
import zlib
def next_object_id(self, offset=None):
    try:
        reference = IndirectReference(max(self.xref_table.keys()) + 1, 0)
    except ValueError:
        reference = IndirectReference(1, 0)
    if offset is not None:
        self.xref_table[reference.object_id] = (offset, 0)
    return reference