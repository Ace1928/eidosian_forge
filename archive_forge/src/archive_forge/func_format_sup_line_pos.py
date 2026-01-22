import re
import html
from paste.util import PySourceColor
def format_sup_line_pos(self, line, column):
    if column:
        return self.emphasize('Line %i, Column %i' % (line, column))
    else:
        return self.emphasize('Line %i' % line)