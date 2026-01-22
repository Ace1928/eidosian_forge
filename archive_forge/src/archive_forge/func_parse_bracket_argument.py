from __future__ import print_function
from __future__ import unicode_literals
import re
import sys
from cmakelang import common
def parse_bracket_argument(text):
    regex = re.compile('^\\[(=*)\\[(.*)\\]\\1\\]$', re.DOTALL)
    match = regex.match(text)
    assert match, 'Failed to match bracket argument pattern in {}'.format(text)
    return ('[' + match.group(1) + '[', match.group(2), ']' + match.group(1) + ']')