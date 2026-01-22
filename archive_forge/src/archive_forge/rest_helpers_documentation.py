import functools
import operator
Flatten a dict into a list of (name,value) tuples.

    The result is suitable for setting query params on an http request.

    .. code-block:: python

        >>> obj = {'a':
        ...         {'b':
        ...           {'c': ['x', 'y', 'z']} },
        ...      'd': 'uvw',
        ...      'e': True, }
        >>> flatten_query_params(obj, strict=True)
        [('a.b.c', 'x'), ('a.b.c', 'y'), ('a.b.c', 'z'), ('d', 'uvw'), ('e', 'true')]

    Note that, as described in
    https://github.com/googleapis/googleapis/blob/48d9fb8c8e287c472af500221c6450ecd45d7d39/google/api/http.proto#L117,
    repeated fields (i.e. list-valued fields) may only contain primitive types (not lists or dicts).
    This is enforced in this function.

    Args:
      obj: a possibly nested dictionary (from json), or None
      strict: a bool, defaulting to False, to enforce that all values in the
              result tuples be strings and, if boolean, lower-cased.

    Returns: a list of tuples, with each tuple having a (possibly) multi-part name
      and a scalar value.

    Raises:
      TypeError if obj is not a dict or None
      ValueError if obj contains a list of non-primitive values.
    