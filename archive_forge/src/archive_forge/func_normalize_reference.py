import re
import sys
from builtins import str, chr
def normalize_reference(string):
    """
        Normalize reference label: collapse internal whitespace
        to single space, remove leading/trailing whitespace, case fold.
        """
    return SMP_RE.sub(_subst_handler, string[1:-1].strip()).translate(XLAT)