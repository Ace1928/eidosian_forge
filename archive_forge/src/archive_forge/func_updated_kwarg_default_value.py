import wrapt
from inspect import signature
from debtcollector import _utils
def updated_kwarg_default_value(name, old_value, new_value, message=None, version=None, stacklevel=3, category=FutureWarning):
    """Decorates a kwarg accepting function to change the default value"""
    prefix = _KWARG_UPDATED_PREFIX_TPL % (name, new_value)
    postfix = _KWARG_UPDATED_POSTFIX_TPL % old_value
    out_message = _utils.generate_message(prefix, postfix=postfix, message=message, version=version)

    def decorator(f):
        sig = signature(f)
        varnames = list(sig.parameters.keys())

        @wrapt.decorator
        def wrapper(wrapped, instance, args, kwargs):
            explicit_params = set(varnames[:len(args)] + list(kwargs.keys()))
            allparams = set(varnames)
            default_params = set(allparams - explicit_params)
            if name in default_params:
                _utils.deprecation(out_message, stacklevel=stacklevel, category=category)
            return wrapped(*args, **kwargs)
        return wrapper(f)
    return decorator