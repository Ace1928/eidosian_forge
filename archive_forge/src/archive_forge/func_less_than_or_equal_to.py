import operator
from typing import Any, Callable
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.matcher import Matcher
def less_than_or_equal_to(value: Any) -> Matcher[Any]:
    """Matches if object is less than or equal to a given value.

    :param value: The value to compare against.

    """
    return OrderingComparison(value, operator.le, 'less than or equal to')