import functools
import sys
import testtools
from testtools.matchers import Is
from fixtures import MonkeyPatch, TestWithFixtures
def test_patch_classmethod_with_staticmethod(self):
    oldmethod = C.foo_cls
    oldmethod_inst = C().foo_cls
    fixture = MonkeyPatch('fixtures.tests._fixtures.test_monkeypatch.C.foo_cls', D.bar_static_args)
    with fixture:
        cls, = C.foo_cls()
        self.expectThat(cls, Is(C))
        cls, = C().foo_cls()
        self.expectThat(cls, Is(C))
    self._check_restored_static_or_class_method(oldmethod, oldmethod_inst, C, 'foo_cls')