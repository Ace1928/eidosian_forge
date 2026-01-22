import functools
import inspect
def get_func_full_args(func):
    """
    Return a list of (argument name, default value) tuples. If the argument
    does not have a default value, omit it in the tuple. Arguments such as
    *args and **kwargs are also included.
    """
    params = _get_callable_parameters(func)
    args = []
    for param in params:
        name = param.name
        if name == 'self':
            continue
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            name = '*' + name
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            name = '**' + name
        if param.default != inspect.Parameter.empty:
            args.append((name, param.default))
        else:
            args.append((name,))
    return args