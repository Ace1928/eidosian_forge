from __future__ import unicode_literals
import bisect
import re
import six
import string
import weakref
from six.moves import range, map
from .selection import SelectionType, SelectionState, PasteMode
from .clipboard import ClipboardData
def find_boundaries_of_current_word(self, WORD=False, include_leading_whitespace=False, include_trailing_whitespace=False):
    """
        Return the relative boundaries (startpos, endpos) of the current word under the
        cursor. (This is at the current line, because line boundaries obviously
        don't belong to any word.)
        If not on a word, this returns (0,0)
        """
    text_before_cursor = self.current_line_before_cursor[::-1]
    text_after_cursor = self.current_line_after_cursor

    def get_regex(include_whitespace):
        return {(False, False): _FIND_CURRENT_WORD_RE, (False, True): _FIND_CURRENT_WORD_INCLUDE_TRAILING_WHITESPACE_RE, (True, False): _FIND_CURRENT_BIG_WORD_RE, (True, True): _FIND_CURRENT_BIG_WORD_INCLUDE_TRAILING_WHITESPACE_RE}[WORD, include_whitespace]
    match_before = get_regex(include_leading_whitespace).search(text_before_cursor)
    match_after = get_regex(include_trailing_whitespace).search(text_after_cursor)
    if not WORD and match_before and match_after:
        c1 = self.text[self.cursor_position - 1]
        c2 = self.text[self.cursor_position]
        alphabet = string.ascii_letters + '0123456789_'
        if (c1 in alphabet) != (c2 in alphabet):
            match_before = None
    return (-match_before.end(1) if match_before else 0, match_after.end(1) if match_after else 0)