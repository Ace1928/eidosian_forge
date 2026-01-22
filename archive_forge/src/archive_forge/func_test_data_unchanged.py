import sys
import dill
def test_data_unchanged():
    FooS = dill.copy(Foo)
    try:
        res = FooS().data
    except Exception:
        e = sys.exc_info()[1]
        raise AssertionError(str(e))
    else:
        assert res == 1