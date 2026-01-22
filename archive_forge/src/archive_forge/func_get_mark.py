from __future__ import absolute_import
import codecs
from ruamel.yaml.error import YAMLError, FileMark, StringMark, YAMLStreamError
from ruamel.yaml.compat import text_type, binary_type, PY3, UNICODE_SIZE
from ruamel.yaml.util import RegExp
def get_mark(self):
    if self.stream is None:
        return StringMark(self.name, self.index, self.line, self.column, self.buffer, self.pointer)
    else:
        return FileMark(self.name, self.index, self.line, self.column)