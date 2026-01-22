from __future__ import division
import contextlib
import json
import numbers
from jsonschema import _utils, _validators
from jsonschema.compat import (
from jsonschema.exceptions import ErrorTree  # Backwards compat  # noqa: F401
from jsonschema.exceptions import RefResolutionError, SchemaError, UnknownType
def resolve_fragment(self, document, fragment):
    """
        Resolve a ``fragment`` within the referenced ``document``.

        Arguments:

            document:

                The referrant document

            fragment (str):

                a URI fragment to resolve within it

        """
    fragment = fragment.lstrip(u'/')
    parts = unquote(fragment).split(u'/') if fragment else []
    for part in parts:
        part = part.replace(u'~1', u'/').replace(u'~0', u'~')
        if isinstance(document, Sequence):
            try:
                part = int(part)
            except ValueError:
                pass
        try:
            document = document[part]
        except (TypeError, LookupError):
            raise RefResolutionError('Unresolvable JSON pointer: %r' % fragment)
    return document