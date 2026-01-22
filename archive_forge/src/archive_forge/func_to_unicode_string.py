from __future__ import annotations
import re
from fractions import Fraction
def to_unicode_string(self):
    """Unicode string with proper sub and superscripts. Note that this works only
        with systems where the sub and superscripts are pure integers.
        """
    str_ = self.to_latex_string()
    for m in re.finditer('\\$_\\{(\\d+)\\}\\$', str_):
        s1 = m.group()
        s2 = [SUBSCRIPT_UNICODE[s] for s in m.group(1)]
        str_ = str_.replace(s1, ''.join(s2))
    for m in re.finditer('\\$\\^\\{([\\d\\+\\-]+)\\}\\$', str_):
        s1 = m.group()
        s2 = [SUPERSCRIPT_UNICODE[s] for s in m.group(1)]
        str_ = str_.replace(s1, ''.join(s2))
    return str_