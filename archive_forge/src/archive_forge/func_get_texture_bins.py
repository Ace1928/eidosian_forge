import os
import sys
import zipfile
import weakref
from io import BytesIO
import pyglet
def get_texture_bins(self):
    """Get a list of texture bins in use.

        This is useful for debugging and profiling only.

        :rtype: list
        :return: List of :py:class:`~pyglet.image.atlas.TextureBin`
        """
    self._require_index()
    return list(self._texture_atlas_bins.values())