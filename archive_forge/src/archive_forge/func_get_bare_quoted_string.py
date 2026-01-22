import re
import sys
import urllib   # For urllib.parse.unquote
from string import hexdigits
from operator import itemgetter
from email import _encoded_words as _ew
from email import errors
from email import utils
def get_bare_quoted_string(value):
    """bare-quoted-string = DQUOTE *([FWS] qcontent) [FWS] DQUOTE

    A quoted-string without the leading or trailing white space.  Its
    value is the text between the quote marks, with whitespace
    preserved and quoted pairs decoded.
    """
    if value[0] != '"':
        raise errors.HeaderParseError('expected \'"\' but found \'{}\''.format(value))
    bare_quoted_string = BareQuotedString()
    value = value[1:]
    if value and value[0] == '"':
        token, value = get_qcontent(value)
        bare_quoted_string.append(token)
    while value and value[0] != '"':
        if value[0] in WSP:
            token, value = get_fws(value)
        elif value[:2] == '=?':
            valid_ew = False
            try:
                token, value = get_encoded_word(value)
                bare_quoted_string.defects.append(errors.InvalidHeaderDefect('encoded word inside quoted string'))
                valid_ew = True
            except errors.HeaderParseError:
                token, value = get_qcontent(value)
            if valid_ew and len(bare_quoted_string) > 1:
                if bare_quoted_string[-1].token_type == 'fws' and bare_quoted_string[-2].token_type == 'encoded-word':
                    bare_quoted_string[-1] = EWWhiteSpaceTerminal(bare_quoted_string[-1], 'fws')
        else:
            token, value = get_qcontent(value)
        bare_quoted_string.append(token)
    if not value:
        bare_quoted_string.defects.append(errors.InvalidHeaderDefect('end of header inside quoted string'))
        return (bare_quoted_string, value)
    return (bare_quoted_string, value[1:])