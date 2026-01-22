import dill
def test_method_with_internal_import_should_work():
    import re
    back_fn = dill.loads(dill.dumps(get_fun_with_internal_import()))
    import inspect
    if hasattr(inspect, 'getclosurevars'):
        vars = inspect.getclosurevars(back_fn)
        assert vars.globals == {}
        assert vars.nonlocals == {}
    assert back_fn() == re.compile('$')
    assert '__builtins__' in back_fn.__globals__