import dill
def quad_factory(a=1, b=1, c=0):

    def dec(f):

        def func(*args, **kwds):
            fx = f(*args, **kwds)
            return a * fx ** 2 + b * fx + c
        return func
    return dec