import functools
import sys
import testtools
from testtools.matchers import Is
from fixtures import MonkeyPatch, TestWithFixtures
def test_patch_unboundmethod_with_boundmethod(self):
    oldmethod = C.foo
    d = D()
    fixture = MonkeyPatch('fixtures.tests._fixtures.test_monkeypatch.C.foo', d.bar_two_args)
    with fixture:
        c = C()
        slf, target_self = c.foo()
        self.expectThat(slf, Is(d))
        self.expectThat(target_self, Is(c))
    self.assertEqual(oldmethod, C.foo)