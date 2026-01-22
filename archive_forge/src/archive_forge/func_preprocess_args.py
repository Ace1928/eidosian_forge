import decorator
from moviepy.tools import cvsecs
def preprocess_args(fun, varnames):
    """ Applies fun to variables in varnames before launching the function """

    def wrapper(f, *a, **kw):
        if hasattr(f, 'func_code'):
            func_code = f.func_code
        else:
            func_code = f.__code__
        names = func_code.co_varnames
        new_a = [fun(arg) if name in varnames else arg for arg, name in zip(a, names)]
        new_kw = {k: fun(v) if k in varnames else v for k, v in kw.items()}
        return f(*new_a, **new_kw)
    return decorator.decorator(wrapper)