from __future__ import print_function
import re
import sys
import time
from pygments.filter import apply_filters, Filter
from pygments.filters import get_filter_by_name
from pygments.token import Error, Text, Other, _TokenType
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
from pygments.regexopt import regex_opt
def get_tokendefs(cls):
    """
        Merge tokens from superclasses in MRO order, returning a single tokendef
        dictionary.

        Any state that is not defined by a subclass will be inherited
        automatically.  States that *are* defined by subclasses will, by
        default, override that state in the superclass.  If a subclass wishes to
        inherit definitions from a superclass, it can use the special value
        "inherit", which will cause the superclass' state definition to be
        included at that point in the state.
        """
    tokens = {}
    inheritable = {}
    for c in cls.__mro__:
        toks = c.__dict__.get('tokens', {})
        for state, items in iteritems(toks):
            curitems = tokens.get(state)
            if curitems is None:
                tokens[state] = items
                try:
                    inherit_ndx = items.index(inherit)
                except ValueError:
                    continue
                inheritable[state] = inherit_ndx
                continue
            inherit_ndx = inheritable.pop(state, None)
            if inherit_ndx is None:
                continue
            curitems[inherit_ndx:inherit_ndx + 1] = items
            try:
                new_inh_ndx = items.index(inherit)
            except ValueError:
                pass
            else:
                inheritable[state] = inherit_ndx + new_inh_ndx
    return tokens