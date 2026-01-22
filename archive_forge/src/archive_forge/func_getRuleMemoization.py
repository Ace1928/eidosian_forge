from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import inspect
import sys
from antlr3 import runtime_version, runtime_version_str
from antlr3.compat import set, frozenset, reversed
from antlr3.constants import DEFAULT_CHANNEL, HIDDEN_CHANNEL, EOF, \
from antlr3.exceptions import RecognitionException, MismatchedTokenException, \
from antlr3.tokens import CommonToken, EOF_TOKEN, SKIP_TOKEN
import six
from six import unichr
def getRuleMemoization(self, ruleIndex, ruleStartIndex):
    """
        Given a rule number and a start token index number, return
        MEMO_RULE_UNKNOWN if the rule has not parsed input starting from
        start index.  If this rule has parsed input starting from the
        start index before, then return where the rule stopped parsing.
        It returns the index of the last token matched by the rule.
        """
    if ruleIndex not in self._state.ruleMemo:
        self._state.ruleMemo[ruleIndex] = {}
    return self._state.ruleMemo[ruleIndex].get(ruleStartIndex, self.MEMO_RULE_UNKNOWN)