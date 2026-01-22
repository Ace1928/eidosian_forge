import sys
import re
def comment_replace(matchobj):
    return matchobj.group(1) + matchobj.group(2) + '#' * len(matchobj.group(3))