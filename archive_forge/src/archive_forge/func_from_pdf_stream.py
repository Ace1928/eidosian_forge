from __future__ import annotations
import calendar
import codecs
import collections
import mmap
import os
import re
import time
import zlib
@classmethod
def from_pdf_stream(cls, data):
    return cls(PdfParser.interpret_name(data))