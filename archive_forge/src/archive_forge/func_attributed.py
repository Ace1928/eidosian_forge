import os
import sys
import zipfile
import weakref
from io import BytesIO
import pyglet
def attributed(self, name):
    """Load an attributed text document.

        See `pyglet.text.formats.attributed` for details on this format.

        :Parameters:
            `name` : str
                Filename of the attribute text resource to load.

        :rtype: `FormattedDocument`
        """
    self._require_index()
    file = self.file(name)
    return pyglet.text.load(name, file, 'text/vnd.pyglet-attributed')