import functools
import sys
import testtools
from testtools.matchers import Is
from fixtures import MonkeyPatch, TestWithFixtures
def test_patch_staticmethod_with_staticmethod(self):
    oldmethod = C.foo_static
    oldmethod_inst = C().foo_static
    fixture = MonkeyPatch('fixtures.tests._fixtures.test_monkeypatch.C.foo_static', D.bar_static)
    with fixture:
        C.foo_static()
        C().foo_static()
    self._check_restored_static_or_class_method(oldmethod, oldmethod_inst, C, 'foo_static')