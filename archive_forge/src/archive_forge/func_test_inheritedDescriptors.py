import os
from typing import Dict, Mapping, Sequence
from hamcrest import assert_that, equal_to, not_
from hypothesis import given
from hypothesis.strategies import dictionaries, integers, lists
from twisted.python.systemd import ListenFDs
from twisted.trial.unittest import SynchronousTestCase
from .strategies import systemdDescriptorNames
@given(lists(integers(min_value=0, max_value=10), unique=True))
def test_inheritedDescriptors(self, descriptors: Sequence[int]) -> None:
    """
        L{ListenFDs.inheritedDescriptors} returns a copy of the inherited
        descriptors list.
        """
    names = tuple(map(str, descriptors))
    fds = ListenFDs(descriptors, names)
    fdsCopy = fds.inheritedDescriptors()
    assert_that(descriptors, equal_to(fdsCopy))
    fdsCopy.append(1)
    assert_that(descriptors, not_(equal_to(fdsCopy)))