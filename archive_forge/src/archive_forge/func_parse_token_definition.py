from __future__ import absolute_import
from . import Actions
from . import DFA
from . import Errors
from . import Machines
from . import Regexps
def parse_token_definition(self, token_spec):
    if not isinstance(token_spec, tuple):
        raise Errors.InvalidToken('Token definition is not a tuple')
    if len(token_spec) != 2:
        raise Errors.InvalidToken('Wrong number of items in token definition')
    pattern, action = token_spec
    if not isinstance(pattern, Regexps.RE):
        raise Errors.InvalidToken('Pattern is not an RE instance')
    return (pattern, action)