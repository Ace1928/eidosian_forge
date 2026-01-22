import os
import posixpath
import re
import stat
import threading
from mako import exceptions
from mako import util
from mako.template import Template
def put_template(self, uri, template):
    """Place a new :class:`.Template` object into this
        :class:`.TemplateLookup`, based on the given
        :class:`.Template` object.

        """
    self._collection[uri] = template