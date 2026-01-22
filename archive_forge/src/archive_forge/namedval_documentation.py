from pyasn1 import error
Create named values object.

    The |NamedValues| object represents a collection of string names
    associated with numeric IDs. These objects are used for giving
    names to otherwise numerical values.

    |NamedValues| objects are immutable and duck-type Python
    :class:`dict` object mapping ID to name and vice-versa.

    Parameters
    ----------
    \*args: variable number of two-element :py:class:`tuple`

        name: :py:class:`str`
            Value label

        value: :py:class:`int`
            Numeric value

    Keyword Args
    ------------
    name: :py:class:`str`
        Value label

    value: :py:class:`int`
        Numeric value

    Examples
    --------

    .. code-block:: pycon

        >>> nv = NamedValues('a', 'b', ('c', 0), d=1)
        >>> nv
        >>> {'c': 0, 'd': 1, 'a': 2, 'b': 3}
        >>> nv[0]
        'c'
        >>> nv['a']
        2
    