import functools
import sys
import testtools
from testtools.matchers import Is
from fixtures import MonkeyPatch, TestWithFixtures
def test_patch_unboundmethod_with_staticmethod(self):
    oldmethod = C.foo
    fixture = MonkeyPatch('fixtures.tests._fixtures.test_monkeypatch.C.foo', D.bar_static_args)
    with fixture:
        target_self, arg = INST_C.foo(1)
        self.expectThat(target_self, Is(INST_C))
        self.assertEqual(1, arg)
    self.assertEqual(oldmethod, C.foo)