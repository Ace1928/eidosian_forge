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
def is_enumerated_list_item(self, ordinal, sequence, format):
    """
        Check validity based on the ordinal value and the second line.

        Return true if the ordinal is valid and the second line is blank,
        indented, or starts with the next enumerator or an auto-enumerator.
        """
    if ordinal is None:
        return None
    try:
        next_line = self.state_machine.next_line()
    except EOFError:
        self.state_machine.previous_line()
        return 1
    else:
        self.state_machine.previous_line()
    if not next_line[:1].strip():
        return 1
    result = self.make_enumerator(ordinal + 1, sequence, format)
    if result:
        next_enumerator, auto_enumerator = result
        try:
            if next_line.startswith(next_enumerator) or next_line.startswith(auto_enumerator):
                return 1
        except TypeError:
            pass
    return None