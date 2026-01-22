from unittest import mock
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy import _parser
from oslo_policy.tests import base
def test_parse_state_meta(self):

    class FakeState(metaclass=_parser.ParseStateMeta):

        @_parser.reducer('a', 'b', 'c')
        @_parser.reducer('d', 'e', 'f')
        def reduce1(self):
            pass

        @_parser.reducer('g', 'h', 'i')
        def reduce2(self):
            pass
    self.assertTrue(hasattr(FakeState, 'reducers'))
    for reduction, reducer in FakeState.reducers:
        if reduction == ['a', 'b', 'c'] or reduction == ['d', 'e', 'f']:
            self.assertEqual('reduce1', reducer)
        elif reduction == ['g', 'h', 'i']:
            self.assertEqual('reduce2', reducer)
        else:
            self.fail('Unrecognized reducer discovered')