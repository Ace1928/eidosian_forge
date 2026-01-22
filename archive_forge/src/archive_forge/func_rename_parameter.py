import contextlib
import functools
import inspect
import math
import warnings
def rename_parameter(since, old, new, func=None):
    """
    Decorator indicating that parameter *old* of *func* is renamed to *new*.

    The actual implementation of *func* should use *new*, not *old*.  If *old*
    is passed to *func*, a DeprecationWarning is emitted, and its value is
    used, even if *new* is also passed by keyword (this is to simplify pyplot
    wrapper functions, which always pass *new* explicitly to the Axes method).
    If *new* is also passed but positionally, a TypeError will be raised by the
    underlying function during argument binding.

    Examples
    --------
    ::

        @_api.rename_parameter("3.1", "bad_name", "good_name")
        def func(good_name): ...
    """
    decorator = functools.partial(rename_parameter, since, old, new)
    if func is None:
        return decorator
    signature = inspect.signature(func)
    assert old not in signature.parameters, f'Matplotlib internal error: {old!r} cannot be a parameter for {func.__name__}()'
    assert new in signature.parameters, f'Matplotlib internal error: {new!r} must be a parameter for {func.__name__}()'

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if old in kwargs:
            warn_deprecated(since, message=f'The {old!r} parameter of {func.__name__}() has been renamed {new!r} since Matplotlib {since}; support for the old name will be dropped %(removal)s.')
            kwargs[new] = kwargs.pop(old)
        return func(*args, **kwargs)
    DECORATORS[wrapper] = decorator
    return wrapper