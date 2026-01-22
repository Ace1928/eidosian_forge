import os
import pkg_resources
from urllib.parse import quote
import string
import inspect
def sub_catcher(filename, vars, func, *args, **kw):
    """
    Run a substitution, returning the value.  If an error occurs, show
    the filename.  If the error is a NameError, show the variables.
    """
    try:
        return func(*args, **kw)
    except SkipTemplate as e:
        print('Skipping file %s' % filename)
        if str(e):
            print(str(e))
        raise
    except Exception as e:
        print('Error in file %s:' % filename)
        if isinstance(e, NameError):
            for name, value in sorted(vars.items()):
                print('%s = %r' % (name, value))
        raise