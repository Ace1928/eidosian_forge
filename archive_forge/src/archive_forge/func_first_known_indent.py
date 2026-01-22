import sys
import re
import types
import unicodedata
from docutils import utils
from docutils.utils.error_reporting import ErrorOutput
def first_known_indent(self, match, context, next_state):
    """
        Handle an indented text block (first line's indent known).

        Extend or override in subclasses.

        Recursively run the registered state machine for known-indent indented
        blocks (`self.known_indent_sm`). The indent is the length of the
        match, ``match.end()``.
        """
    indented, line_offset, blank_finish = self.state_machine.get_first_known_indented(match.end())
    sm = self.known_indent_sm(debug=self.debug, **self.known_indent_sm_kwargs)
    results = sm.run(indented, input_offset=line_offset)
    return (context, next_state, results)