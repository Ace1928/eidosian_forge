def skip_doctest(f):
    """Decorator - mark a function or method for skipping its doctest.

    This decorator allows you to mark a function whose docstring you wish to
    omit from testing, while preserving the docstring for introspection, help,
    etc."""
    f.__skip_doctest__ = True
    return f