from typing import MutableSequence, Optional, Sequence, TypeVar, Union, cast
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.helpers.wrap_matcher import wrap_matcher
from hamcrest.core.matcher import Matcher
def isnotsurplus(self, item: T) -> bool:
    if not self.matchers:
        if self.mismatch_description:
            self.mismatch_description.append_text('not matched: ').append_description_of(item)
        return False
    return True