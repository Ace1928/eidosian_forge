from __future__ import absolute_import
import re
import operator
import sys
def handle_func_not(next, token):
    """
    not(...)
    """
    name, predicate = parse_func(next, token)

    def select(result):
        for node in result:
            if _get_first_or_none(predicate([node])) is None:
                yield node
    return select