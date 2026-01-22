from typing import Any, Optional
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.matcher import Matcher
Matches anything.

    :param description: Optional string used to describe this matcher.

    This matcher always evaluates to ``True``. Specify this in composite
    matchers when the value of a particular element is unimportant.

    