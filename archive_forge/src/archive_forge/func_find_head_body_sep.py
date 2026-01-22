import re
import sys
from docutils import DataError
from docutils.utils import strip_combining_chars
def find_head_body_sep(self):
    """Look for a head/body row separator line; store the line index."""
    for i in range(len(self.block)):
        line = self.block[i]
        if self.head_body_separator_pat.match(line):
            if self.head_body_sep:
                raise TableMarkupError('Multiple head/body row separators (table lines %s and %s); only one allowed.' % (self.head_body_sep + 1, i + 1), offset=i)
            else:
                self.head_body_sep = i
                self.block[i] = line.replace('=', '-')
    if self.head_body_sep == 0 or self.head_body_sep == len(self.block) - 1:
        raise TableMarkupError('The head/body row separator may not be the first or last line of the table.', offset=i)