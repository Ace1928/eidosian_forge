import sys
import re
import types
import unicodedata
from docutils import utils
from docutils.utils.error_reporting import ErrorOutput
def insert_input(self, input_lines, source):
    self.input_lines.insert(self.line_offset + 1, '', source='internal padding after ' + source, offset=len(input_lines))
    self.input_lines.insert(self.line_offset + 1, '', source='internal padding before ' + source, offset=-1)
    self.input_lines.insert(self.line_offset + 2, StringList(input_lines, source))