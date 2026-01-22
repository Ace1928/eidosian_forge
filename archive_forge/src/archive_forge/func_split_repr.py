from __future__ import annotations
from inspect import getfullargspec
def split_repr(self):
    """
    Return a helpful description of the object using
    self._repr_words() and self._repr_attrs() to add
    to the description.  This function may be used by
    adding code to your class like this:

    >>> class Foo(object):
    ...     __repr__ = split_repr
    ...     def _repr_words(self):
    ...         return ["words", "here"]
    ...     def _repr_attrs(self):
    ...         return {'attrs': "appear too"}
    >>> Foo()
    <Foo words here attrs='appear too'>
    >>> class Bar(Foo):
    ...     def _repr_words(self):
    ...         return Foo._repr_words(self) + ["too"]
    ...     def _repr_attrs(self):
    ...         return dict(Foo._repr_attrs(self), barttr=42)
    >>> Bar()
    <Bar words here too attrs='appear too' barttr=42>
    """
    alist = sorted(((str(k), normalize_repr(v)) for k, v in self._repr_attrs().items()))
    words = self._repr_words()
    if not words and (not alist):
        return super(self.__class__, self).__repr__()
    if words and alist:
        words.append('')
    return f'<{self.__class__.__name__} {' '.join(words) + ' '.join([f'{k}={v}' for k, v in alist])}>'