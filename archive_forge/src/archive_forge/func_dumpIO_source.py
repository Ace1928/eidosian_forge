import contextlib
def dumpIO_source(object, **kwds):
    """write object source to a buffer (instead of dill.dump)
Loads by with dill.temp.loadIO_source.  Returns the buffer object.

    >>> f = lambda x:x**2
    >>> pyfile = dill.temp.dumpIO_source(f, alias='_f')
    >>> _f = dill.temp.loadIO_source(pyfile)
    >>> _f(4)
    16

Optional kwds:
    If 'alias' is specified, the object will be renamed to the given string.
    """
    from .source import importable, getname
    from io import BytesIO as StringIO
    alias = kwds.pop('alias', '')
    name = str(alias) or getname(object)
    name = '\n#NAME: %s\n' % name
    file = StringIO()
    file.write(b(''.join([importable(object, alias=alias), name])))
    file.flush()
    return file