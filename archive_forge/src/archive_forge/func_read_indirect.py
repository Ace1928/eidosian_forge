from __future__ import annotations
import calendar
import codecs
import collections
import mmap
import os
import re
import time
import zlib
def read_indirect(self, ref, max_nesting=-1):
    offset, generation = self.xref_table[ref[0]]
    check_format_condition(generation == ref[1], f'expected to find generation {ref[1]} for object ID {ref[0]} in xref table, instead found generation {generation} at offset {offset}')
    value = self.get_value(self.buf, offset + self.start_offset, expect_indirect=IndirectReference(*ref), max_nesting=max_nesting)[0]
    self.cached_objects[ref] = value
    return value