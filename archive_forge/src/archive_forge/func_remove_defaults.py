from __future__ import annotations
from inspect import getfullargspec
def remove_defaults(d, fn):
    """
    Remove keys in d that are set to the default values from
    fn.  This method is used to unclutter the _repr_attrs()
    return value.

    d will be modified by this function.

    Returns d.

    >>> class Foo(object):
    ...     def __init__(self, a=1, b=2):
    ...         self.values = a, b
    ...     __repr__ = split_repr
    ...     def _repr_words(self):
    ...         return ["object"]
    ...     def _repr_attrs(self):
    ...         d = dict(a=self.values[0], b=self.values[1])
    ...         return remove_defaults(d, Foo.__init__)
    >>> Foo(42, 100)
    <Foo object a=42 b=100>
    >>> Foo(10, 2)
    <Foo object a=10>
    >>> Foo()
    <Foo object>
    """
    args, varargs, varkw, defaults, _, _, _ = getfullargspec(fn)
    if varkw:
        del args[-1]
    if varargs:
        del args[-1]
    ddict = dict(zip(args[len(args) - len(defaults):], defaults))
    for k in list(d.keys()):
        if k in ddict and ddict[k] == d[k]:
            del d[k]
    return d