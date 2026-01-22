from __future__ import annotations
import calendar
import codecs
import collections
import mmap
import os
import re
import time
import zlib
def read_trailer(self):
    search_start_offset = len(self.buf) - 16384
    if search_start_offset < self.start_offset:
        search_start_offset = self.start_offset
    m = self.re_trailer_end.search(self.buf, search_start_offset)
    check_format_condition(m, 'trailer end not found')
    last_match = m
    while m:
        last_match = m
        m = self.re_trailer_end.search(self.buf, m.start() + 16)
    if not m:
        m = last_match
    trailer_data = m.group(1)
    self.last_xref_section_offset = int(m.group(2))
    self.trailer_dict = self.interpret_trailer(trailer_data)
    self.xref_table = XrefTable()
    self.read_xref_table(xref_section_offset=self.last_xref_section_offset)
    if b'Prev' in self.trailer_dict:
        self.read_prev_trailer(self.trailer_dict[b'Prev'])