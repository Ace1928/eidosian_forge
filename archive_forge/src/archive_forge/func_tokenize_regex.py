from __future__ import unicode_literals
import re
def tokenize_regex(input):
    """
    Takes a string, representing a regular expression as input, and tokenizes
    it.

    :param input: string, representing a regular expression.
    :returns: List of tokens.
    """
    p = re.compile('^(\n        \\(\\?P\\<[a-zA-Z0-9_-]+\\>  | # Start of named group.\n        \\(\\?#[^)]*\\)             | # Comment\n        \\(\\?=                    | # Start of lookahead assertion\n        \\(\\?!                    | # Start of negative lookahead assertion\n        \\(\\?<=                   | # If preceded by.\n        \\(\\?<                    | # If not preceded by.\n        \\(?:                     | # Start of group. (non capturing.)\n        \\(                       | # Start of group.\n        \\(?[iLmsux]              | # Flags.\n        \\(?P=[a-zA-Z]+\\)         | # Back reference to named group\n        \\)                       | # End of group.\n        \\{[^{}]*\\}               | # Repetition\n        \\*\\? | \\+\\? | \\?\\?\\      | # Non greedy repetition.\n        \\* | \\+ | \\?             | # Repetition\n        \\#.*\\n                   | # Comment\n        \\\\. |\n\n        # Character group.\n        \\[\n            ( [^\\]\\\\]  |  \\\\.)*\n        \\]                  |\n\n        [^(){}]             |\n        .\n    )', re.VERBOSE)
    tokens = []
    while input:
        m = p.match(input)
        if m:
            token, input = (input[:m.end()], input[m.end():])
            if not token.isspace():
                tokens.append(token)
        else:
            raise Exception('Could not tokenize input regex.')
    return tokens