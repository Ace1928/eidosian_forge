from __future__ import unicode_literals
import re
from six.moves import range
from .regex_parser import Any, Sequence, Regex, Variable, Repeat, Lookahead
from .regex_parser import parse_regex, tokenize_regex
def end_nodes(self):
    """
        Yields `MatchVariable` instances for all the nodes having their end
        position at the end of the input string.
        """
    for varname, reg in self._nodes_to_regs():
        if reg[1] == len(self.string):
            value = self._unescape(varname, self.string[reg[0]:reg[1]])
            yield MatchVariable(varname, value, (reg[0], reg[1]))