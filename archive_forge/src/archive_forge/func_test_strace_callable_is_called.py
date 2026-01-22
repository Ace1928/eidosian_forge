import threading
from breezy import strace, tests
from breezy.strace import StraceResult, strace_detailed
from breezy.tests.features import strace_feature
def test_strace_callable_is_called(self):
    self._check_threads()
    output = []

    def function(positional, *args, **kwargs):
        output.append((positional, args, kwargs))
    self.strace_detailed_or_skip(function, ['a', 'b'], {'c': 'c'}, follow_children=False)
    self.assertEqual([('a', ('b',), {'c': 'c'})], output)