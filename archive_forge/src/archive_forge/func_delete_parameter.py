import contextlib
import functools
import inspect
import math
import warnings
def delete_parameter(since, name, func=None, **kwargs):
    """
    Decorator indicating that parameter *name* of *func* is being deprecated.

    The actual implementation of *func* should keep the *name* parameter in its
    signature, or accept a ``**kwargs`` argument (through which *name* would be
    passed).

    Parameters that come after the deprecated parameter effectively become
    keyword-only (as they cannot be passed positionally without triggering the
    DeprecationWarning on the deprecated parameter), and should be marked as
    such after the deprecation period has passed and the deprecated parameter
    is removed.

    Parameters other than *since*, *name*, and *func* are keyword-only and
    forwarded to `.warn_deprecated`.

    Examples
    --------
    ::

        @_api.delete_parameter("3.1", "unused")
        def func(used_arg, other_arg, unused, more_args): ...
    """
    decorator = functools.partial(delete_parameter, since, name, **kwargs)
    if func is None:
        return decorator
    signature = inspect.signature(func)
    kwargs_name = next((param.name for param in signature.parameters.values() if param.kind == inspect.Parameter.VAR_KEYWORD), None)
    if name in signature.parameters:
        kind = signature.parameters[name].kind
        is_varargs = kind is inspect.Parameter.VAR_POSITIONAL
        is_varkwargs = kind is inspect.Parameter.VAR_KEYWORD
        if not is_varargs and (not is_varkwargs):
            name_idx = math.inf if kind is inspect.Parameter.KEYWORD_ONLY else [*signature.parameters].index(name)
            func.__signature__ = signature = signature.replace(parameters=[param.replace(default=_deprecated_parameter) if param.name == name else param for param in signature.parameters.values()])
        else:
            name_idx = -1
    else:
        is_varargs = is_varkwargs = False
        name_idx = math.inf
        assert kwargs_name, f'Matplotlib internal error: {name!r} must be a parameter for {func.__name__}()'
    addendum = kwargs.pop('addendum', None)

    @functools.wraps(func)
    def wrapper(*inner_args, **inner_kwargs):
        if len(inner_args) <= name_idx and name not in inner_kwargs:
            return func(*inner_args, **inner_kwargs)
        arguments = signature.bind(*inner_args, **inner_kwargs).arguments
        if is_varargs and arguments.get(name):
            warn_deprecated(since, message=f'Additional positional arguments to {func.__name__}() are deprecated since %(since)s and support for them will be removed %(removal)s.')
        elif is_varkwargs and arguments.get(name):
            warn_deprecated(since, message=f'Additional keyword arguments to {func.__name__}() are deprecated since %(since)s and support for them will be removed %(removal)s.')
        elif any((name in d and d[name] != _deprecated_parameter for d in [arguments, arguments.get(kwargs_name, {})])):
            deprecation_addendum = f'If any parameter follows {name!r}, they should be passed as keyword, not positionally.'
            warn_deprecated(since, name=repr(name), obj_type=f'parameter of {func.__name__}()', addendum=addendum + ' ' + deprecation_addendum if addendum else deprecation_addendum, **kwargs)
        return func(*inner_args, **inner_kwargs)
    DECORATORS[wrapper] = decorator
    return wrapper