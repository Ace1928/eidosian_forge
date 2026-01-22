import inspect
import sys
def register_shim(name, default):
    """
    A easier to understand and more compact way of "softly" defining a function.
    These two pieces of code are equivalent:

    if _dill.OLD3X:
        def _create_class():
            ...
    _create_class = register_shim('_create_class', types.new_class)

    if _dill.OLD3X:
        @move_to(_dill)
        def _create_class():
            ...
    _create_class = Getattr(_dill, '_create_class', types.new_class)

    Intuitively, it creates a function or object in the versions of dill/python
    that require special reimplementations, and use a core library or default
    implementation if that function or object does not exist.
    """
    func = globals().get(name)
    if func is not None:
        _dill.__dict__[name] = func
        func.__module__ = _dill.__name__
    if default is Getattr.NO_DEFAULT:
        reduction = (getattr, (_dill, name))
    else:
        reduction = (getattr, (_dill, name, default))
    return Reduce(*reduction, is_callable=callable(default))