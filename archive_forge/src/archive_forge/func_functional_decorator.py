import functools
def functional_decorator():

    def decorator(f):

        def functional_wrapper(*args, **kwargs):
            return f(*args, **kwargs)
        return functional_wrapper
    return decorator