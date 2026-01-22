import functools
import sys
import testtools
from testtools.matchers import Is
from fixtures import MonkeyPatch, TestWithFixtures
def test_patch_c_foo_with_instance_d_bar_self_referential(self):
    oldmethod = C.foo
    oldmethod_inst = C().foo
    fixture = MonkeyPatch('fixtures.tests._fixtures.test_monkeypatch.C.foo', D().bar_self_referential)
    with fixture:
        C().foo()
    self.assertEqual(oldmethod, C.foo)
    self.assertEqual(oldmethod_inst.__code__, C().foo.__code__)