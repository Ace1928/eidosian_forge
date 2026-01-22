from __future__ import absolute_import
from __future__ import print_function
import sys
from ruamel.yaml.error import YAMLError, YAMLStreamError
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.compat import utf8, text_type, PY2, nprint, dbg, DBG_EVENT, \
def expect_document_end(self):
    if isinstance(self.event, DocumentEndEvent):
        self.write_indent()
        if self.event.explicit:
            self.write_indicator(u'...', True)
            self.write_indent()
        self.flush_stream()
        self.state = self.expect_document_start
    else:
        raise EmitterError('expected DocumentEndEvent, but got %s' % (self.event,))