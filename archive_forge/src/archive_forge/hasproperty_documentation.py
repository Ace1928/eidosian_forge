from typing import Any, Mapping, TypeVar, Union, overload
from hamcrest import described_as
from hamcrest.core import anything
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.core.allof import AllOf
from hamcrest.core.description import Description
from hamcrest.core.helpers.wrap_matcher import wrap_matcher as wrap_shortcut
from hamcrest.core.matcher import Matcher
from hamcrest.core.string_description import StringDescription
Matches if an object has properties satisfying all of a dictionary
    of string property names and corresponding value matchers.

    :param matcher_dict: A dictionary mapping keys to associated value matchers,
        or to expected values for
        :py:func:`~hamcrest.core.core.isequal.equal_to` matching.

    Note that the keys must be actual keys, not matchers. Any value argument
    that is not a matcher is implicitly wrapped in an
    :py:func:`~hamcrest.core.core.isequal.equal_to` matcher to check for
    equality.

    Examples::

        has_properties({'foo':equal_to(1), 'bar':equal_to(2)})
        has_properties({'foo':1, 'bar':2})

    ``has_properties`` also accepts a list of keyword arguments:

    .. function:: has_properties(keyword1=value_matcher1[, keyword2=value_matcher2[, ...]])

    :param keyword1: A keyword to look up.
    :param valueMatcher1: The matcher to satisfy for the value, or an expected
        value for :py:func:`~hamcrest.core.core.isequal.equal_to` matching.

    Examples::

        has_properties(foo=equal_to(1), bar=equal_to(2))
        has_properties(foo=1, bar=2)

    Finally, ``has_properties`` also accepts a list of alternating keys and their
    value matchers:

    .. function:: has_properties(key1, value_matcher1[, ...])

    :param key1: A key (not a matcher) to look up.
    :param valueMatcher1: The matcher to satisfy for the value, or an expected
        value for :py:func:`~hamcrest.core.core.isequal.equal_to` matching.

    Examples::

        has_properties('foo', equal_to(1), 'bar', equal_to(2))
        has_properties('foo', 1, 'bar', 2)

    