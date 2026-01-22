from typing import Type, TypeVar, Union
from hamcrest.core.base_matcher import Matcher
from hamcrest.core.core.isequal import equal_to
Wraps argument in a matcher, if necessary.

    :returns: the argument as-is if it is already a matcher, otherwise wrapped
        in an :py:func:`~hamcrest.core.core.isequal.equal_to` matcher.

    