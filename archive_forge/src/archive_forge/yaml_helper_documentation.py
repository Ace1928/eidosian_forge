from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
Given a yaml object, yield all objects found by following a path.

  Given a yaml object, read each field in the path and return the object
  found at the end. If a field has a list value, follow the path for each
  object in the list.

  E.g.
  >>> X = {'A': {'B': [{'C': {'D': 1}}, {'C': {'D': 2}}]}}
  >>> sorted(list(GetAll(X, path=('A', 'B', 'C', 'D'))))
  [1, 2]

  Args:
    obj: A dictionary representing a yaml dictionary
    path: A list of strings representing fields to follow.

  Yields:
    Values that are found by following the given path.
  