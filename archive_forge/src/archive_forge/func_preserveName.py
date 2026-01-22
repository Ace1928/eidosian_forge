from types import CodeType as code, FunctionType as function
def preserveName(f):
    """
    Preserve the name of the given function on the decorated function.
    """

    def decorator(decorated):
        return copyfunction(decorated, dict(name=f.__name__), dict(name=f.__name__))
    return decorator