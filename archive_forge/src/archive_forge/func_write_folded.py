from __future__ import absolute_import
from __future__ import print_function
import sys
from ruamel.yaml.error import YAMLError, YAMLStreamError
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.compat import utf8, text_type, PY2, nprint, dbg, DBG_EVENT, \
def write_folded(self, text):
    hints, _indent, _indicator = self.determine_block_hints(text)
    self.write_indicator(u'>' + hints, True)
    if _indicator == u'+':
        self.open_ended = True
    self.write_line_break()
    leading_space = True
    spaces = False
    breaks = True
    start = end = 0
    while end <= len(text):
        ch = None
        if end < len(text):
            ch = text[end]
        if breaks:
            if ch is None or ch not in u'\n\x85\u2028\u2029\x07':
                if not leading_space and ch is not None and (ch != u' ') and (text[start] == u'\n'):
                    self.write_line_break()
                leading_space = ch == u' '
                for br in text[start:end]:
                    if br == u'\n':
                        self.write_line_break()
                    else:
                        self.write_line_break(br)
                if ch is not None:
                    self.write_indent()
                start = end
        elif spaces:
            if ch != u' ':
                if start + 1 == end and self.column > self.best_width:
                    self.write_indent()
                else:
                    data = text[start:end]
                    self.column += len(data)
                    if bool(self.encoding):
                        data = data.encode(self.encoding)
                    self.stream.write(data)
                start = end
        elif ch is None or ch in u' \n\x85\u2028\u2029\x07':
            data = text[start:end]
            self.column += len(data)
            if bool(self.encoding):
                data = data.encode(self.encoding)
            self.stream.write(data)
            if ch == u'\x07':
                if end < len(text) - 1 and (not text[end + 2].isspace()):
                    self.write_line_break()
                    self.write_indent()
                    end += 2
                else:
                    raise EmitterError('unexcpected fold indicator \\a before space')
            if ch is None:
                self.write_line_break()
            start = end
        if ch is not None:
            breaks = ch in u'\n\x85\u2028\u2029'
            spaces = ch == u' '
        end += 1