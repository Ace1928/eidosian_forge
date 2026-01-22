from typing import TypeVar
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.matcher import Matcher
def same_instance(obj: T) -> Matcher[T]:
    """Matches if evaluated object is the same instance as a given object.

    :param obj: The object to compare against as the expected value.

    This matcher invokes the ``is`` identity operator to determine if the
    evaluated object is the the same object as ``obj``.

    """
    return IsSame(obj)