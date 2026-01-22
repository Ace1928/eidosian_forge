from __future__ import absolute_import
import re
import operator
import sys
def parse_func(next, token):
    name = token[1]
    token = next()
    if token[0] != '(':
        raise ValueError("Expected '(' after function name '%s'" % name)
    predicate = handle_predicate(next, token)
    return (name, predicate)