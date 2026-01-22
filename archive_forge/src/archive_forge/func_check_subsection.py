import sys
import re
from types import FunctionType, MethodType
from docutils import nodes, statemachine, utils
from docutils import ApplicationError, DataError
from docutils.statemachine import StateMachineWS, StateWS
from docutils.nodes import fully_normalize_name as normalize_name
from docutils.nodes import whitespace_normalize_name
import docutils.parsers.rst
from docutils.parsers.rst import directives, languages, tableparser, roles
from docutils.parsers.rst.languages import en as _fallback_language_module
from docutils.utils import escape2null, unescape, column_width
from docutils.utils import punctuation_chars, roman, urischemes
from docutils.utils import split_escaped_whitespace
def check_subsection(self, source, style, lineno):
    """
        Check for a valid subsection header.  Return 1 (true) or None (false).

        When a new section is reached that isn't a subsection of the current
        section, back up the line count (use ``previous_line(-x)``), then
        ``raise EOFError``.  The current StateMachine will finish, then the
        calling StateMachine can re-examine the title.  This will work its way
        back up the calling chain until the correct section level isreached.

        @@@ Alternative: Evaluate the title, store the title info & level, and
        back up the chain until that level is reached.  Store in memo? Or
        return in results?

        :Exception: `EOFError` when a sibling or supersection encountered.
        """
    memo = self.memo
    title_styles = memo.title_styles
    mylevel = memo.section_level
    try:
        level = title_styles.index(style) + 1
    except ValueError:
        if len(title_styles) == memo.section_level:
            title_styles.append(style)
            return 1
        else:
            self.parent += self.title_inconsistent(source, lineno)
            return None
    if level <= mylevel:
        memo.section_level = level
        if len(style) == 2:
            memo.section_bubble_up_kludge = True
        self.state_machine.previous_line(len(style) + 1)
        raise EOFError
    if level == mylevel + 1:
        return 1
    else:
        self.parent += self.title_inconsistent(source, lineno)
        return None