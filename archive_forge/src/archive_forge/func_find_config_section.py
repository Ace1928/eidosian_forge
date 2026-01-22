from configparser import ConfigParser
import os
import re
import sys
from urllib.parse import unquote
from paste.deploy.util import fix_call, importlib_metadata, lookup_object
def find_config_section(self, object_type, name=None):
    """
        Return the section name with the given name prefix (following the
        same pattern as ``protocol_desc`` in ``config``.  It must have the
        given name, or for ``'main'`` an empty name is allowed.  The
        prefix must be followed by a ``:``.

        Case is *not* ignored.
        """
    possible = []
    for name_options in object_type.config_prefixes:
        for name_prefix in name_options:
            found = self._find_sections(self.parser.sections(), name_prefix, name)
            if found:
                possible.extend(found)
                break
    if not possible:
        raise LookupError('No section %r (prefixed by %s) found in config %s' % (name, ' or '.join(map(repr, _flatten(object_type.config_prefixes))), self.filename))
    if len(possible) > 1:
        raise LookupError('Ambiguous section names %r for section %r (prefixed by %s) found in config %s' % (possible, name, ' or '.join(map(repr, _flatten(object_type.config_prefixes))), self.filename))
    return possible[0]