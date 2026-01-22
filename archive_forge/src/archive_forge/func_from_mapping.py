from __future__ import annotations
import errno
import json
import os
import types
import typing as t
from werkzeug.utils import import_string
def from_mapping(self, mapping: t.Mapping[str, t.Any] | None=None, **kwargs: t.Any) -> bool:
    """Updates the config like :meth:`update` ignoring items with
        non-upper keys.

        :return: Always returns ``True``.

        .. versionadded:: 0.11
        """
    mappings: dict[str, t.Any] = {}
    if mapping is not None:
        mappings.update(mapping)
    mappings.update(kwargs)
    for key, value in mappings.items():
        if key.isupper():
            self[key] = value
    return True