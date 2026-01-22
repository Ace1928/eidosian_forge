import os
import posixpath
import re
import stat
import threading
from mako import exceptions
from mako import util
from mako.template import Template
def put_string(self, uri, text):
    """Place a new :class:`.Template` object into this
        :class:`.TemplateLookup`, based on the given string of
        ``text``.

        """
    self._collection[uri] = Template(text, lookup=self, uri=uri, **self.template_args)