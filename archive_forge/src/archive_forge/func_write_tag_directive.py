from __future__ import absolute_import
from __future__ import print_function
import sys
from ruamel.yaml.error import YAMLError, YAMLStreamError
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.compat import utf8, text_type, PY2, nprint, dbg, DBG_EVENT, \
def write_tag_directive(self, handle_text, prefix_text):
    data = u'%%TAG %s %s' % (handle_text, prefix_text)
    if self.encoding:
        data = data.encode(self.encoding)
    self.stream.write(data)
    self.write_line_break()