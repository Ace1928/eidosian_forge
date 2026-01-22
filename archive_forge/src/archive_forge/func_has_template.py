import os
import posixpath
import re
import stat
import threading
from mako import exceptions
from mako import util
from mako.template import Template
def has_template(self, uri):
    """Return ``True`` if this :class:`.TemplateLookup` is
        capable of returning a :class:`.Template` object for the
        given ``uri``.

        :param uri: String URI of the template to be resolved.

        """
    try:
        self.get_template(uri)
        return True
    except exceptions.TemplateLookupException:
        return False