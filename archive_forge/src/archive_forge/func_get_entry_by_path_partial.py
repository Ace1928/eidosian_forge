from collections import deque
from ..lazy_import import lazy_import
from breezy.bzr import (
from .. import errors, lazy_regex, osutils, trace
from .static_tuple import StaticTuple
def get_entry_by_path_partial(self, relpath):
    """Like get_entry_by_path, but return TreeReference objects.

        :param relpath: Path to resolve, either as string with / as separators,
            or as list of elements.
        :return: tuple with ie, resolved elements and elements left to resolve
        """
    if isinstance(relpath, str):
        names = osutils.splitpath(relpath)
    else:
        names = relpath
    try:
        parent = self.root
    except errors.NoSuchId:
        return (None, None, None)
    if parent is None:
        return (None, None, None)
    for i, f in enumerate(names):
        try:
            children = getattr(parent, 'children', None)
            if children is None:
                return (None, None, None)
            cie = children[f]
            if cie.kind == 'tree-reference':
                return (cie, names[:i + 1], names[i + 1:])
            parent = cie
        except KeyError:
            return (None, None, None)
    return (parent, names, [])