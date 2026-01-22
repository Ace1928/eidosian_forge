from functools import wraps
from .sympify import SympifyError, sympify
def sympify_method_args(cls):
    """Decorator for a class with methods that sympify arguments.

    Explanation
    ===========

    The sympify_method_args decorator is to be used with the sympify_return
    decorator for automatic sympification of method arguments. This is
    intended for the common idiom of writing a class like :

    Examples
    ========

    >>> from sympy import Basic, SympifyError, S
    >>> from sympy.core.sympify import _sympify

    >>> class MyTuple(Basic):
    ...     def __add__(self, other):
    ...         try:
    ...             other = _sympify(other)
    ...         except SympifyError:
    ...             return NotImplemented
    ...         if not isinstance(other, MyTuple):
    ...             return NotImplemented
    ...         return MyTuple(*(self.args + other.args))

    >>> MyTuple(S(1), S(2)) + MyTuple(S(3), S(4))
    MyTuple(1, 2, 3, 4)

    In the above it is important that we return NotImplemented when other is
    not sympifiable and also when the sympified result is not of the expected
    type. This allows the MyTuple class to be used cooperatively with other
    classes that overload __add__ and want to do something else in combination
    with instance of Tuple.

    Using this decorator the above can be written as

    >>> from sympy.core.decorators import sympify_method_args, sympify_return

    >>> @sympify_method_args
    ... class MyTuple(Basic):
    ...     @sympify_return([('other', 'MyTuple')], NotImplemented)
    ...     def __add__(self, other):
    ...          return MyTuple(*(self.args + other.args))

    >>> MyTuple(S(1), S(2)) + MyTuple(S(3), S(4))
    MyTuple(1, 2, 3, 4)

    The idea here is that the decorators take care of the boiler-plate code
    for making this happen in each method that potentially needs to accept
    unsympified arguments. Then the body of e.g. the __add__ method can be
    written without needing to worry about calling _sympify or checking the
    type of the resulting object.

    The parameters for sympify_return are a list of tuples of the form
    (parameter_name, expected_type) and the value to return (e.g.
    NotImplemented). The expected_type parameter can be a type e.g. Tuple or a
    string 'Tuple'. Using a string is useful for specifying a Type within its
    class body (as in the above example).

    Notes: Currently sympify_return only works for methods that take a single
    argument (not including self). Specifying an expected_type as a string
    only works for the class in which the method is defined.
    """
    for attrname, obj in cls.__dict__.items():
        if isinstance(obj, _SympifyWrapper):
            setattr(cls, attrname, obj.make_wrapped(cls))
    return cls