from __future__ import annotations
import calendar
import codecs
import collections
import mmap
import os
import re
import time
import zlib
def start_writing(self):
    self.close_buf()
    self.seek_end()