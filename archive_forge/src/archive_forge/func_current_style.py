import re
from humanfriendly.compat import HTMLParser, StringIO, name2codepoint, unichr
from humanfriendly.text import compact_empty_lines
from humanfriendly.terminal import ANSI_COLOR_CODES, ANSI_RESET, ansi_style
@property
def current_style(self):
    """Get the current style from the top of the stack (a dictionary)."""
    return self.stack[-1] if self.stack else {}