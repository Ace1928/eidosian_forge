from typing import Any, List, Sequence, Tuple, TypeVar
from hamcrest import (
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.core.allof import AllOf
from hamcrest.core.description import Description
from hamcrest.core.matcher import Matcher
from typing_extensions import Protocol
from twisted.python.failure import Failure
def isFailure(**properties: Matcher[object]) -> Matcher[object]:
    """
    Match an instance of L{Failure} with matching attributes.
    """
    return AllOf(instance_of(Failure), has_properties(**properties))