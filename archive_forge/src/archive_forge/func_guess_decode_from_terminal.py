import re
import sys
def guess_decode_from_terminal(text, term):
    """Decode *text* coming from terminal *term*.

    First try the terminal encoding, if given.
    Then try UTF-8.  Then try the preferred locale encoding.
    Fall back to latin-1, which always works.
    """
    if getattr(term, 'encoding', None):
        try:
            text = text.decode(term.encoding)
        except UnicodeDecodeError:
            pass
        else:
            return (text, term.encoding)
    return guess_decode(text)