import functools
import sys
import testtools
from testtools.matchers import Is
from fixtures import MonkeyPatch, TestWithFixtures
def test_double_patch_classmethod(self):
    oldmethod = C.foo_cls
    oldmethod_inst = C().foo_cls
    fixture1 = MonkeyPatch('fixtures.tests._fixtures.test_monkeypatch.C.foo_cls', fake)
    fixture2 = MonkeyPatch('fixtures.tests._fixtures.test_monkeypatch.C.foo_cls', fake2)
    with fixture1:
        with fixture2:
            C.foo_cls()
            C().foo_cls()
    self._check_restored_static_or_class_method(oldmethod, oldmethod_inst, C, 'foo_cls')