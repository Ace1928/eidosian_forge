from __future__ import absolute_import
from __future__ import print_function
import sys
from ruamel.yaml.error import YAMLError, YAMLStreamError
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.compat import utf8, text_type, PY2, nprint, dbg, DBG_EVENT, \
def determine_block_hints(self, text):
    indent = 0
    indicator = u''
    hints = u''
    if text:
        if text[0] in u' \n\x85\u2028\u2029':
            indent = self.best_sequence_indent
            hints += text_type(indent)
        elif self.root_context:
            for end in ['\n---', '\n...']:
                pos = 0
                while True:
                    pos = text.find(end, pos)
                    if pos == -1:
                        break
                    try:
                        if text[pos + 4] in ' \r\n':
                            break
                    except IndexError:
                        pass
                    pos += 1
                if pos > -1:
                    break
            if pos > 0:
                indent = self.best_sequence_indent
        if text[-1] not in u'\n\x85\u2028\u2029':
            indicator = u'-'
        elif len(text) == 1 or text[-2] in u'\n\x85\u2028\u2029':
            indicator = u'+'
    hints += indicator
    return (hints, indent, indicator)