import re
import sys
from os.path import expanduser
import patiencediff
from . import terminal, trace
from .commands import get_cmd_object
from .patches import (ContextLine, Hunk, HunkLine, InsertLine, RemoveLine,
def colorstring(self, type, item, bad_ws_match):
    color = self.colors[type]
    if color is not None:
        if self.check_style and bad_ws_match:
            item.contents = ''.join((terminal.colorstring(txt, color, bcol) for txt, bcol in ((bad_ws_match.group(1).expandtabs(), self.colors['leadingtabs']), (bad_ws_match.group(2)[0:self.max_line_len], None), (bad_ws_match.group(2)[self.max_line_len:], self.colors['longline']), (bad_ws_match.group(3), self.colors['trailingspace'])))) + bad_ws_match.group(4)
        if not isinstance(item, bytes):
            item = item.as_bytes()
        string = terminal.colorstring(item, color)
    else:
        string = str(item)
    self.target.write(string)