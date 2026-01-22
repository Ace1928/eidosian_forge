from __future__ import print_function
from __future__ import unicode_literals
import re
import sys
from cmakelang import common
def parse_bracket_comment(text):
    prefix, content, suffix = parse_bracket_argument(text[1:])
    return ('#' + prefix, content, suffix)