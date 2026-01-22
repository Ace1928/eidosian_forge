from __future__ import absolute_import
from __future__ import print_function
import sys
from ruamel.yaml.error import YAMLError, YAMLStreamError
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.compat import utf8, text_type, PY2, nprint, dbg, DBG_EVENT, \
def write_plain(self, text, split=True):
    if self.root_context:
        if self.requested_indent is not None:
            self.write_line_break()
            if self.requested_indent != 0:
                self.write_indent()
        else:
            self.open_ended = True
    if not text:
        return
    if not self.whitespace:
        data = u' '
        self.column += len(data)
        if self.encoding:
            data = data.encode(self.encoding)
        self.stream.write(data)
    self.whitespace = False
    self.indention = False
    spaces = False
    breaks = False
    start = end = 0
    while end <= len(text):
        ch = None
        if end < len(text):
            ch = text[end]
        if spaces:
            if ch != u' ':
                if start + 1 == end and self.column > self.best_width and split:
                    self.write_indent()
                    self.whitespace = False
                    self.indention = False
                else:
                    data = text[start:end]
                    self.column += len(data)
                    if self.encoding:
                        data = data.encode(self.encoding)
                    self.stream.write(data)
                start = end
        elif breaks:
            if ch not in u'\n\x85\u2028\u2029':
                if text[start] == u'\n':
                    self.write_line_break()
                for br in text[start:end]:
                    if br == u'\n':
                        self.write_line_break()
                    else:
                        self.write_line_break(br)
                self.write_indent()
                self.whitespace = False
                self.indention = False
                start = end
        elif ch is None or ch in u' \n\x85\u2028\u2029':
            data = text[start:end]
            self.column += len(data)
            if self.encoding:
                data = data.encode(self.encoding)
            try:
                self.stream.write(data)
            except:
                sys.stdout.write(repr(data) + '\n')
                raise
            start = end
        if ch is not None:
            spaces = ch == u' '
            breaks = ch in u'\n\x85\u2028\u2029'
        end += 1