from __future__ import absolute_import
from __future__ import print_function
import sys
from ruamel.yaml.error import YAMLError, YAMLStreamError
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.compat import utf8, text_type, PY2, nprint, dbg, DBG_EVENT, \
def expect_document_start(self, first=False):
    if isinstance(self.event, DocumentStartEvent):
        if (self.event.version or self.event.tags) and self.open_ended:
            self.write_indicator(u'...', True)
            self.write_indent()
        if self.event.version:
            version_text = self.prepare_version(self.event.version)
            self.write_version_directive(version_text)
        self.tag_prefixes = self.DEFAULT_TAG_PREFIXES.copy()
        if self.event.tags:
            handles = sorted(self.event.tags.keys())
            for handle in handles:
                prefix = self.event.tags[handle]
                self.tag_prefixes[prefix] = handle
                handle_text = self.prepare_tag_handle(handle)
                prefix_text = self.prepare_tag_prefix(prefix)
                self.write_tag_directive(handle_text, prefix_text)
        implicit = first and (not self.event.explicit) and (not self.canonical) and (not self.event.version) and (not self.event.tags) and (not self.check_empty_document())
        if not implicit:
            self.write_indent()
            self.write_indicator(u'---', True)
            if self.canonical:
                self.write_indent()
        self.state = self.expect_document_root
    elif isinstance(self.event, StreamEndEvent):
        if self.open_ended:
            self.write_indicator(u'...', True)
            self.write_indent()
        self.write_stream_end()
        self.state = self.expect_nothing
    else:
        raise EmitterError('expected DocumentStartEvent, but got %s' % (self.event,))