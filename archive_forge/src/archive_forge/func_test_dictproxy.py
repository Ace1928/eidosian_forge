import dill
import weakref
def test_dictproxy():
    from dill._dill import DictProxyType
    try:
        m = DictProxyType({'foo': 'bar'})
    except Exception:
        m = type.__dict__
    mp = dill.copy(m)
    assert mp.items() == m.items()