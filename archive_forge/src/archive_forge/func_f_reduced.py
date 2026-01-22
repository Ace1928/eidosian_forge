from functools import wraps
def f_reduced(x):
    if hasattr(x, '__iter__'):
        return list(map(f_reduced, x))
    else:
        if is_arg:
            args[n] = x
        else:
            kwargs[n] = x
        return f(*args, **kwargs)