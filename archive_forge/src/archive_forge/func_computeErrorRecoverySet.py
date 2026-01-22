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
def computeErrorRecoverySet(self):
    """
        Compute the error recovery set for the current rule.  During
        rule invocation, the parser pushes the set of tokens that can
        follow that rule reference on the stack; this amounts to
        computing FIRST of what follows the rule reference in the
        enclosing rule. This local follow set only includes tokens
        from within the rule; i.e., the FIRST computation done by
        ANTLR stops at the end of a rule.

        EXAMPLE

        When you find a "no viable alt exception", the input is not
        consistent with any of the alternatives for rule r.  The best
        thing to do is to consume tokens until you see something that
        can legally follow a call to r *or* any rule that called r.
        You don't want the exact set of viable next tokens because the
        input might just be missing a token--you might consume the
        rest of the input looking for one of the missing tokens.

        Consider grammar:

        a : '[' b ']'
          | '(' b ')'
          ;
        b : c '^' INT ;
        c : ID
          | INT
          ;

        At each rule invocation, the set of tokens that could follow
        that rule is pushed on a stack.  Here are the various "local"
        follow sets:

        FOLLOW(b1_in_a) = FIRST(']') = ']'
        FOLLOW(b2_in_a) = FIRST(')') = ')'
        FOLLOW(c_in_b) = FIRST('^') = '^'

        Upon erroneous input "[]", the call chain is

        a -> b -> c

        and, hence, the follow context stack is:

        depth  local follow set     after call to rule
          0         \\<EOF>                    a (from main())
          1          ']'                     b
          3          '^'                     c

        Notice that ')' is not included, because b would have to have
        been called from a different context in rule a for ')' to be
        included.

        For error recovery, we cannot consider FOLLOW(c)
        (context-sensitive or otherwise).  We need the combined set of
        all context-sensitive FOLLOW sets--the set of all tokens that
        could follow any reference in the call chain.  We need to
        resync to one of those tokens.  Note that FOLLOW(c)='^' and if
        we resync'd to that token, we'd consume until EOF.  We need to
        sync to context-sensitive FOLLOWs for a, b, and c: {']','^'}.
        In this case, for input "[]", LA(1) is in this set so we would
        not consume anything and after printing an error rule c would
        return normally.  It would not find the required '^' though.
        At this point, it gets a mismatched token error and throws an
        exception (since LA(1) is not in the viable following token
        set).  The rule exception handler tries to recover, but finds
        the same recovery set and doesn't consume anything.  Rule b
        exits normally returning to rule a.  Now it finds the ']' (and
        with the successful match exits errorRecovery mode).

        So, you cna see that the parser walks up call chain looking
        for the token that was a member of the recovery set.

        Errors are not generated in errorRecovery mode.

        ANTLR's error recovery mechanism is based upon original ideas:

        "Algorithms + Data Structures = Programs" by Niklaus Wirth

        and

        "A note on error recovery in recursive descent parsers":
        http://portal.acm.org/citation.cfm?id=947902.947905

        Later, Josef Grosch had some good ideas:

        "Efficient and Comfortable Error Recovery in Recursive Descent
        Parsers":
        ftp://www.cocolab.com/products/cocktail/doca4.ps/ell.ps.zip

        Like Grosch I implemented local FOLLOW sets that are combined
        at run-time upon error to avoid overhead during parsing.
        """
    return self.combineFollows(False)