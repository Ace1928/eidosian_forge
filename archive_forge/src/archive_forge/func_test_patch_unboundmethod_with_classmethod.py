import functools
import sys
import testtools
from testtools.matchers import Is
from fixtures import MonkeyPatch, TestWithFixtures
def test_patch_unboundmethod_with_classmethod(self):
    oldmethod = C.foo
    fixture = MonkeyPatch('fixtures.tests._fixtures.test_monkeypatch.C.foo', D.bar_cls_args)
    with fixture:
        c = C()
        cls, target_self, arg = c.foo(1)
        self.expectThat(cls, Is(D))
        self.expectThat(target_self, Is(c))
        self.assertEqual(1, arg)
    self.assertEqual(oldmethod, C.foo)