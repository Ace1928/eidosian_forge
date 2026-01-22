from typing import Any, Hashable, Mapping, Optional, TypeVar, Union, overload
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.helpers.wrap_matcher import wrap_matcher
from hamcrest.core.matcher import Matcher
def describe_keyvalue(self, index: K, value: V, description: Description) -> None:
    """Describes key-value pair at given index."""
    description.append_description_of(index).append_text(': ').append_description_of(value)