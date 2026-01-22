from __future__ import absolute_import
from __future__ import print_function
import sys
from ruamel.yaml.error import YAMLError, YAMLStreamError
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.compat import utf8, text_type, PY2, nprint, dbg, DBG_EVENT, \
def write_version_directive(self, version_text):
    data = u'%%YAML %s' % version_text
    if self.encoding:
        data = data.encode(self.encoding)
    self.stream.write(data)
    self.write_line_break()