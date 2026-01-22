import functools
import dill
import sys
def test_issue_510():

    class Foo:

        def __init__(self):

            def f2(self):
                return self
            self.f2 = f2.__get__(self)
    import dill, pickletools
    f = Foo()
    f1 = dill.copy(f)
    assert f1.f2() is f1