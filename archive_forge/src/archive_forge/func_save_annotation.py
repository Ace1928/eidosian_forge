from __future__ import absolute_import
import os
import os.path
import re
import codecs
import textwrap
from datetime import datetime
from functools import partial
from collections import defaultdict
from xml.sax.saxutils import escape as html_escape
from . import Version
from .Code import CCodeWriter
from .. import Utils
def save_annotation(self, source_filename, target_filename, coverage_xml=None):
    with Utils.open_source_file(source_filename) as f:
        code = f.read()
    generated_code = self.code.get(source_filename, {})
    c_file = Utils.decode_filename(os.path.basename(target_filename))
    html_filename = os.path.splitext(target_filename)[0] + '.html'
    with codecs.open(html_filename, 'w', encoding='UTF-8') as out_buffer:
        out_buffer.write(self._save_annotation(code, generated_code, c_file, source_filename, coverage_xml))