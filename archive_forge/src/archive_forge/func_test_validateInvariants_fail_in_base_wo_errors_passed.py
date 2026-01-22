import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_validateInvariants_fail_in_base_wo_errors_passed(self):
    from zope.interface.exceptions import Invalid
    _passable_called_with = []

    def _passable(*args, **kw):
        _passable_called_with.append((args, kw))
        return True
    _fail_called_with = []

    def _fail(*args, **kw):
        _fail_called_with.append((args, kw))
        raise Invalid
    base = self._makeOne('IBase')
    derived = self._makeOne('IDerived', (base,))
    obj = object()
    base.setTaggedValue('invariants', [_fail])
    derived.setTaggedValue('invariants', [_passable])
    self.assertRaises(Invalid, derived.validateInvariants, obj)
    self.assertEqual(_passable_called_with, [((obj,), {})])
    self.assertEqual(_fail_called_with, [((obj,), {})])