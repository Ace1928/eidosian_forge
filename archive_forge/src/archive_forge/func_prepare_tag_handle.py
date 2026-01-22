from __future__ import absolute_import
from __future__ import print_function
import sys
from ruamel.yaml.error import YAMLError, YAMLStreamError
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.compat import utf8, text_type, PY2, nprint, dbg, DBG_EVENT, \
def prepare_tag_handle(self, handle):
    if not handle:
        raise EmitterError('tag handle must not be empty')
    if handle[0] != u'!' or handle[-1] != u'!':
        raise EmitterError("tag handle must start and end with '!': %r" % utf8(handle))
    for ch in handle[1:-1]:
        if not (u'0' <= ch <= u'9' or u'A' <= ch <= u'Z' or u'a' <= ch <= u'z' or (ch in u'-_')):
            raise EmitterError('invalid character %r in the tag handle: %r' % (utf8(ch), utf8(handle)))
    return handle