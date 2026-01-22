from __future__ import annotations
import calendar
import codecs
import collections
import mmap
import os
import re
import time
import zlib
def seek_end(self):
    self.f.seek(0, os.SEEK_END)