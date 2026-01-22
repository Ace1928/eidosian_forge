from __future__ import annotations
import decimal
import re
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Callable
def to_gettext(rule: Mapping[str, str] | Iterable[tuple[str, str]] | PluralRule) -> str:
    """The plural rule as gettext expression.  The gettext expression is
    technically limited to integers and returns indices rather than tags.

    >>> to_gettext({'one': 'n is 1', 'two': 'n is 2'})
    'nplurals=3; plural=((n == 1) ? 0 : (n == 2) ? 1 : 2);'

    :param rule: the rules as list or dict, or a `PluralRule` object
    :raise RuleError: if the expression is malformed
    """
    rule = PluralRule.parse(rule)
    used_tags = rule.tags | {_fallback_tag}
    _compile = _GettextCompiler().compile
    _get_index = [tag for tag in _plural_tags if tag in used_tags].index
    result = [f'nplurals={len(used_tags)}; plural=(']
    for tag, ast in rule.abstract:
        result.append(f'{_compile(ast)} ? {_get_index(tag)} : ')
    result.append(f'{_get_index(_fallback_tag)});')
    return ''.join(result)