import functools
import sys
import testtools
from testtools.matchers import Is
from fixtures import MonkeyPatch, TestWithFixtures
def test_double_patch_staticmethod(self):
    oldmethod = C.foo_static
    oldmethod_inst = C().foo_static
    fixture1 = MonkeyPatch('fixtures.tests._fixtures.test_monkeypatch.C.foo_static', fake_no_args)
    fixture2 = MonkeyPatch('fixtures.tests._fixtures.test_monkeypatch.C.foo_static', fake_static)
    with fixture1:
        with fixture2:
            C.foo_static()
            C().foo_static()
    self._check_restored_static_or_class_method(oldmethod, oldmethod_inst, C, 'foo_static')