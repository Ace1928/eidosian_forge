from __future__ import absolute_import
from __future__ import print_function
import sys
from ruamel.yaml.error import YAMLError, YAMLStreamError
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.compat import utf8, text_type, PY2, nprint, dbg, DBG_EVENT, \
def expect_stream_start(self):
    if isinstance(self.event, StreamStartEvent):
        if PY2:
            if self.event.encoding and (not getattr(self.stream, 'encoding', None)):
                self.encoding = self.event.encoding
        elif self.event.encoding and (not hasattr(self.stream, 'encoding')):
            self.encoding = self.event.encoding
        self.write_stream_start()
        self.state = self.expect_first_document_start
    else:
        raise EmitterError('expected StreamStartEvent, but got %s' % (self.event,))