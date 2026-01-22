from typing import Any, List, Sequence, Tuple, TypeVar
from hamcrest import (
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.core.allof import AllOf
from hamcrest.core.description import Description
from hamcrest.core.matcher import Matcher
from typing_extensions import Protocol
from twisted.python.failure import Failure
def similarFrame(functionName: str, fileName: str) -> Matcher[Sequence[Tuple[str, str, int, List[object], List[object]]]]:
    """
    Match a tuple representation of a frame like those used by
    L{twisted.python.failure.Failure}.
    """
    return contains_exactly(equal_to(functionName), contains_string(fileName), instance_of(int), has_length(0), has_length(0))