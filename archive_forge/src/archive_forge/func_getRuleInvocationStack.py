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
def getRuleInvocationStack(self):
    """
        Return List<String> of the rules in your parser instance
        leading up to a call to this method.  You could override if
        you want more details such as the file/line info of where
        in the parser java code a rule is invoked.

        This is very useful for error messages and for context-sensitive
        error recovery.

        You must be careful, if you subclass a generated recognizers.
        The default implementation will only search the module of self
        for rules, but the subclass will not contain any rules.
        You probably want to override this method to look like

        def getRuleInvocationStack(self):
            return self._getRuleInvocationStack(<class>.__module__)

        where <class> is the class of the generated recognizer, e.g.
        the superclass of self.
        """
    return self._getRuleInvocationStack(self.__module__)