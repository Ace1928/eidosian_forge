from __future__ import annotations
import decimal
import re
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Callable
def to_javascript(rule: Mapping[str, str] | Iterable[tuple[str, str]] | PluralRule) -> str:
    """Convert a list/dict of rules or a `PluralRule` object into a JavaScript
    function.  This function depends on no external library:

    >>> to_javascript({'one': 'n is 1'})
    "(function(n) { return (n == 1) ? 'one' : 'other'; })"

    Implementation detail: The function generated will probably evaluate
    expressions involved into range operations multiple times.  This has the
    advantage that external helper functions are not required and is not a
    big performance hit for these simple calculations.

    :param rule: the rules as list or dict, or a `PluralRule` object
    :raise RuleError: if the expression is malformed
    """
    to_js = _JavaScriptCompiler().compile
    result = ['(function(n) { return ']
    for tag, ast in PluralRule.parse(rule).abstract:
        result.append(f'{to_js(ast)} ? {tag!r} : ')
    result.append('%r; })' % _fallback_tag)
    return ''.join(result)