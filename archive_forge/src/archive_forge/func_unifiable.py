from .core import unify, reify  # type: ignore[attr-defined]
from .dispatch import dispatch
def unifiable(cls):
    """ Register standard unify and reify operations on class
    This uses the type and __dict__ or __slots__ attributes to define the
    nature of the term
    See Also:
    >>> # xdoctest: +SKIP
    >>> class A(object):
    ...     def __init__(self, a, b):
    ...         self.a = a
    ...         self.b = b
    >>> unifiable(A)
    <class 'unification.more.A'>
    >>> x = var('x')
    >>> a = A(1, 2)
    >>> b = A(1, x)
    >>> unify(a, b, {})
    {~x: 2}
    """
    _unify.add((cls, cls, dict), unify_object)
    _reify.add((cls, dict), reify_object)
    return cls