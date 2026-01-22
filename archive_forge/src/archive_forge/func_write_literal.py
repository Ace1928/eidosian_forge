from __future__ import absolute_import
from __future__ import print_function
import sys
from ruamel.yaml.error import YAMLError, YAMLStreamError
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.compat import utf8, text_type, PY2, nprint, dbg, DBG_EVENT, \
def write_literal(self, text, comment=None):
    hints, _indent, _indicator = self.determine_block_hints(text)
    self.write_indicator(u'|' + hints, True)
    try:
        comment = comment[1][0]
        if comment:
            self.stream.write(comment)
    except (TypeError, IndexError):
        pass
    if _indicator == u'+':
        self.open_ended = True
    self.write_line_break()
    breaks = True
    start = end = 0
    while end <= len(text):
        ch = None
        if end < len(text):
            ch = text[end]
        if breaks:
            if ch is None or ch not in u'\n\x85\u2028\u2029':
                for br in text[start:end]:
                    if br == u'\n':
                        self.write_line_break()
                    else:
                        self.write_line_break(br)
                if ch is not None:
                    if self.root_context:
                        idnx = self.indent if self.indent is not None else 0
                        self.stream.write(u' ' * (_indent + idnx))
                    else:
                        self.write_indent()
                start = end
        elif ch is None or ch in u'\n\x85\u2028\u2029':
            data = text[start:end]
            if bool(self.encoding):
                data = data.encode(self.encoding)
            self.stream.write(data)
            if ch is None:
                self.write_line_break()
            start = end
        if ch is not None:
            breaks = ch in u'\n\x85\u2028\u2029'
        end += 1