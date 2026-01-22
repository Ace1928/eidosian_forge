from __future__ import absolute_import
import codecs
from ruamel.yaml.error import YAMLError, FileMark, StringMark, YAMLStreamError
from ruamel.yaml.compat import text_type, binary_type, PY3, UNICODE_SIZE
from ruamel.yaml.util import RegExp
def reset_reader(self):
    self.name = None
    self.stream_pointer = 0
    self.eof = True
    self.buffer = ''
    self.pointer = 0
    self.raw_buffer = None
    self.raw_decode = None
    self.encoding = None
    self.index = 0
    self.line = 0
    self.column = 0