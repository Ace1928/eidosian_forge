import os
from typing import Dict, Mapping, Sequence
from hamcrest import assert_that, equal_to, not_
from hypothesis import given
from hypothesis.strategies import dictionaries, integers, lists
from twisted.python.systemd import ListenFDs
from twisted.trial.unittest import SynchronousTestCase
from .strategies import systemdDescriptorNames
@given(lists(systemdDescriptorNames(), min_size=0, max_size=10))
def test_fromEnvironmentEquivalence(self, names: Sequence[str]) -> None:
    """
        The L{ListenFDs} and L{ListenFDs.fromEnvironment} constructors are
        equivalent for their respective representations of the same
        information.

        @param names: The names of the file descriptors to represent as
            inherited in the test environment given to the parser.  The number
            of descriptors represented will equal the length of this list.
        """
    numFDs = len(names)
    descriptors = list(range(ListenFDs._START, ListenFDs._START + numFDs))
    fds = ListenFDs.fromEnvironment({'LISTEN_PID': str(os.getpid()), 'LISTEN_FDS': str(numFDs), 'LISTEN_FDNAMES': ':'.join(names)})
    assert_that(fds, equal_to(ListenFDs(descriptors, tuple(names))))