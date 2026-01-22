from __future__ import print_function
import re
from sqlparse import tokens as T
from sqlparse.compat import string_types, text_type, unicode_compatible
from sqlparse.utils import imt, remove_quotes
def token_next(self, idx, skip_ws=True, skip_cm=False, _reverse=False):
    """Returns the next token relative to *idx*.

        If *skip_ws* is ``True`` (the default) whitespace tokens are ignored.
        If *skip_cm* is ``True`` comments are ignored.
        ``None`` is returned if there's no next token.
        """
    if idx is None:
        return (None, None)
    idx += 1
    funcs = lambda tk: not (skip_ws and tk.is_whitespace or (skip_cm and imt(tk, t=T.Comment, i=Comment)))
    return self._token_matching(funcs, idx, reverse=_reverse)