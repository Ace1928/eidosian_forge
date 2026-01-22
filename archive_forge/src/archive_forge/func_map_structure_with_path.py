from collections import abc as collections_abc
import logging
import sys
from typing import Mapping, Sequence, Text, TypeVar, Union
from .sequence import _is_attrs
from .sequence import _is_namedtuple
from .sequence import _sequence_like
from .sequence import _sorted
def map_structure_with_path(func, *structures, **kwargs):
    """Maps `func` through given structures.

  This is a variant of :func:`~tree.map_structure` which accumulates
  a *path* while mapping through the structures. A path is a tuple of
  indices and/or keys which uniquely identifies the positions of the
  arguments passed to `func`.

  >>> tree.map_structure_with_path(
  ...     lambda path, v: (path, v**2),
  ...     [{"foo": 42}])
  [{'foo': ((0, 'foo'), 1764)}]

  Args:
    func: A callable that accepts a path and as many arguments as there are
      structures.
    *structures: Arbitrarily nested structures of the same layout.
    **kwargs: The only valid keyword argument is `check_types`. If `True`
      (default) the types of components within the structures have to be match,
      e.g. ``tree.map_structure_with_path(func, [1], (1,))`` will raise a
      `TypeError`, otherwise this is not enforced. Note that namedtuples with
      identical name and fields are considered to be the same type.

  Returns:
    A new structure with the same layout as the given ones. If the
    `structures` have components of varying types, the resulting structure
    will use the same types as ``structures[0]``.

  Raises:
    TypeError: If `func` is not callable or if the `structures` do not
      have the same layout.
    TypeError: If `check_types` is `True` and any two `structures`
      differ in the types of their components.
    ValueError: If no structures were given or if a keyword argument other
      than `check_types` is provided.
  """
    return map_structure_with_path_up_to(structures[0], func, *structures, **kwargs)