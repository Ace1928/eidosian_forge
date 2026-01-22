from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import collections
import contextlib
import itertools
import tokenize
from six import StringIO
from pasta.base import formatting as fmt
from pasta.base import fstring_utils
def close_scope(self, node, prefix_attr='prefix', suffix_attr='suffix', trailing_comma=False, single_paren=False):
    """Close a parenthesized scope on the given node, if one is open."""
    if fmt.get(node, prefix_attr) is None:
        fmt.set(node, prefix_attr, '')
    if fmt.get(node, suffix_attr) is None:
        fmt.set(node, suffix_attr, '')
    if not self._parens or node not in self._scope_stack[-1]:
        return
    symbols = {')'}
    if trailing_comma:
        symbols.add(',')
    parsed_to_i = self._i
    parsed_to_loc = prev_loc = self._loc
    encountered_paren = False
    result = ''
    for tok in self.takewhile(lambda t: t.type in FORMATTING_TOKENS or t.src in symbols):
        result += self._space_between(prev_loc, tok.start)
        if tok.src == ')' and single_paren and encountered_paren:
            self.rewind()
            parsed_to_i = self._i
            parsed_to_loc = tok.start
            fmt.append(node, suffix_attr, result)
            break
        result += tok.src
        if tok.src == ')':
            encountered_paren = True
            self._scope_stack.pop()
            fmt.prepend(node, prefix_attr, self._parens.pop())
            fmt.append(node, suffix_attr, result)
            result = ''
            parsed_to_i = self._i
            parsed_to_loc = tok.end
            if not self._parens or node not in self._scope_stack[-1]:
                break
        prev_loc = tok.end
    self._i = parsed_to_i
    self._loc = parsed_to_loc