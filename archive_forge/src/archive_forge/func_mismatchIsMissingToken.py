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
def mismatchIsMissingToken(self, input, follow):
    if follow is None:
        return False
    if EOR_TOKEN_TYPE in follow:
        if len(self._state.following) > 0:
            follow = follow - set([EOR_TOKEN_TYPE])
        viableTokensFollowingThisRule = self.computeContextSensitiveRuleFOLLOW()
        follow = follow | viableTokensFollowingThisRule
    if input.LA(1) in follow or EOR_TOKEN_TYPE in follow:
        return True
    return False