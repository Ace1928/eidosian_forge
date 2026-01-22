import re
from email.errors import HeaderParseError
from email.parser import HeaderParser
from inspect import cleandoc
from django.urls import reverse
from django.utils.regex_helper import _lazy_re_compile
from django.utils.safestring import mark_safe
def replace_metacharacters(pattern):
    """Remove unescaped metacharacters from the pattern."""
    return re.sub('((?:^|(?<!\\\\))(?:\\\\\\\\)*)(\\\\?)([?*+^$]|\\\\[bBAZ])', lambda m: m[1] + m[3] if m[2] else m[1], pattern)