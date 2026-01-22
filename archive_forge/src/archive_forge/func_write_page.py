from __future__ import annotations
import calendar
import codecs
import collections
import mmap
import os
import re
import time
import zlib
def write_page(self, ref, *objs, **dict_obj):
    if isinstance(ref, int):
        ref = self.pages[ref]
    if 'Type' not in dict_obj:
        dict_obj['Type'] = PdfName(b'Page')
    if 'Parent' not in dict_obj:
        dict_obj['Parent'] = self.pages_ref
    return self.write_obj(ref, *objs, **dict_obj)