import sys
import re
def quote_replace(matchobj):
    return '{}{}{}{}'.format(matchobj.group(1), matchobj.group(2), 'x' * len(matchobj.group(3)), matchobj.group(2))