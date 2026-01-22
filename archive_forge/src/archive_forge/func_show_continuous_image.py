from __future__ import absolute_import
import sys
def show_continuous_image(self, size=(6, 1)):
    """
        Embed an image of this continuous color map in the IPython Notebook.

        Parameters
        ----------
        size : tuple of int, optional
            (width, height) of image to make in units of inches.

        """
    if sys.version_info[0] == 2:
        from StringIO import StringIO as BytesIO
    elif sys.version_info[0] == 3:
        from io import BytesIO
    from IPython.display import display
    from IPython.display import Image as ipyImage
    im = BytesIO()
    self._write_image(im, 'continuous', format='png', size=size)
    display(ipyImage(data=im.getvalue(), format='png'))