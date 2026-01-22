from __future__ import absolute_import
from __future__ import print_function
import sys
from ruamel.yaml.error import YAMLError, YAMLStreamError
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.compat import utf8, text_type, PY2, nprint, dbg, DBG_EVENT, \
def write_pre_comment(self, event):
    comments = event.comment[1]
    if comments is None:
        return False
    try:
        start_events = (MappingStartEvent, SequenceStartEvent)
        for comment in comments:
            if isinstance(event, start_events) and getattr(comment, 'pre_done', None):
                continue
            if self.column != 0:
                self.write_line_break()
            self.write_comment(comment, pre=True)
            if isinstance(event, start_events):
                comment.pre_done = True
    except TypeError:
        sys.stdout.write('eventtt {} {}'.format(type(event), event))
        raise
    return True