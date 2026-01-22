def eval_import(s):
    """
    Import a module, or import an object from a module.

    A module name like ``foo.bar:baz()`` can be used, where
    ``foo.bar`` is the module, and ``baz()`` is an expression
    evaluated in the context of that module.  Note this is not safe on
    arbitrary strings because of the eval.
    """
    if ':' not in s:
        return simple_import(s)
    module_name, expr = s.split(':', 1)
    module = import_module(module_name)
    obj = eval(expr, module.__dict__)
    return obj