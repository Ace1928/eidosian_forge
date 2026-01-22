import weakref
from enum import Enum
import threading
from typing import Tuple
import pyglet
from pyglet import gl
from pyglet.gl import gl_info
def get_gl_attributes(self):
    """Return a list of attributes set on this config.

        :rtype: list of tuple (name, value)
        :return: All attributes, with unset attributes having a value of
            ``None``.
        """
    return [(name, getattr(self, name)) for name in self._attribute_names]