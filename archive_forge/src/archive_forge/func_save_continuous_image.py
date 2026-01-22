from __future__ import absolute_import
import sys
def save_continuous_image(self, filename, size=(6, 1), format=None):
    """
        Save an image of this continuous color map to a file.

        Parameters
        ----------
        filename : str
            If `format` is None the format will be inferred from the
            `filename` extension.
        size : tuple of int, optional
            (width, height) of image to make in units of inches.
        format : str, optional
            An image format that will be understood by matplotlib.

        """
    with open(filename, 'wb') as f:
        self._write_image(f, 'continuous', format=filename.split('.')[-1], size=size)