from __future__ import annotations
import re
import warnings
from functools import lru_cache
from typing import TYPE_CHECKING, SupportsInt
import datetime
from collections.abc import Iterable
from babel import localtime
from babel.core import Locale, default_locale, get_global
from babel.localedata import LocaleDataDict
def match_skeleton(skeleton: str, options: Iterable[str], allow_different_fields: bool=False) -> str | None:
    """
    Find the closest match for the given datetime skeleton among the options given.

    This uses the rules outlined in the TR35 document.

    >>> match_skeleton('yMMd', ('yMd', 'yMMMd'))
    'yMd'

    >>> match_skeleton('yMMd', ('jyMMd',), allow_different_fields=True)
    'jyMMd'

    >>> match_skeleton('yMMd', ('qyMMd',), allow_different_fields=False)

    >>> match_skeleton('hmz', ('hmv',))
    'hmv'

    :param skeleton: The skeleton to match
    :type skeleton: str
    :param options: An iterable of other skeletons to match against
    :type options: Iterable[str]
    :return: The closest skeleton match, or if no match was found, None.
    :rtype: str|None
    """
    options = sorted((option for option in options if option))
    if 'z' in skeleton and (not any(('z' in option for option in options))):
        skeleton = skeleton.replace('z', 'v')
    get_input_field_width = dict((t[1] for t in tokenize_pattern(skeleton) if t[0] == 'field')).get
    best_skeleton = None
    best_distance = None
    for option in options:
        get_opt_field_width = dict((t[1] for t in tokenize_pattern(option) if t[0] == 'field')).get
        distance = 0
        for field in PATTERN_CHARS:
            input_width = get_input_field_width(field, 0)
            opt_width = get_opt_field_width(field, 0)
            if input_width == opt_width:
                continue
            if opt_width == 0 or input_width == 0:
                if not allow_different_fields:
                    option = None
                    break
                distance += 4096
            elif field == 'M' and (input_width > 2 and opt_width <= 2 or (input_width <= 2 and opt_width > 2)):
                distance += 256
            else:
                distance += abs(input_width - opt_width)
        if not option:
            continue
        if not best_skeleton or distance < best_distance:
            best_skeleton = option
            best_distance = distance
        if distance == 0:
            break
    return best_skeleton