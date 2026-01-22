import os
from fnmatch import fnmatch
from datetime import datetime
import operator
import re
def color_line(line, foreground=None, background=None):
    match = re.search('^(\\s*)', line)
    return match.group(1) + color_code(foreground, background) + line[match.end():] + color_code()