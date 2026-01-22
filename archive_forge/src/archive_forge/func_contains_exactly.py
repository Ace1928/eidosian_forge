import warnings
from typing import Optional, Sequence, TypeVar, Union
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.helpers.wrap_matcher import wrap_matcher
from hamcrest.core.matcher import Matcher
def contains_exactly(*items: Union[Matcher[T], T]) -> Matcher[Sequence[T]]:
    """Matches if sequence's elements satisfy a given list of matchers, in order.

    :param match1,...: A comma-separated list of matchers.

    This matcher iterates the evaluated sequence and a given list of matchers,
    seeing if each element satisfies its corresponding matcher.

    Any argument that is not a matcher is implicitly wrapped in an
    :py:func:`~hamcrest.core.core.isequal.equal_to` matcher to check for
    equality.

    """
    matchers = []
    for item in items:
        matchers.append(wrap_matcher(item))
    return IsSequenceContainingInOrder(matchers)