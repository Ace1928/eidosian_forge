from __future__ import absolute_import
from __future__ import print_function
import sys
from ruamel.yaml.error import YAMLError, YAMLStreamError
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.compat import utf8, text_type, PY2, nprint, dbg, DBG_EVENT, \
def process_anchor(self, indicator):
    if self.event.anchor is None:
        self.prepared_anchor = None
        return False
    if self.prepared_anchor is None:
        self.prepared_anchor = self.prepare_anchor(self.event.anchor)
    if self.prepared_anchor:
        self.write_indicator(indicator + self.prepared_anchor, True)
    self.prepared_anchor = None
    return True