from __future__ import unicode_literals
import re
from six.moves import range
from .regex_parser import Any, Sequence, Regex, Variable, Repeat, Lookahead
from .regex_parser import parse_regex, tokenize_regex
def match_prefix(self, string):
    """
        Do a partial match of the string with the grammar. The returned
        :class:`Match` instance can contain multiple representations of the
        match. This will never return `None`. If it doesn't match at all, the "trailing input"
        part will capture all of the input.

        :param string: The input string.
        """
    for patterns in [self._re_prefix, self._re_prefix_with_trailing_input]:
        matches = [(r, r.match(string)) for r in patterns]
        matches = [(r, m) for r, m in matches if m]
        if matches != []:
            return Match(string, matches, self._group_names_to_nodes, self.unescape_funcs)