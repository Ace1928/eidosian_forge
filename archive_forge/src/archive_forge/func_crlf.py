import codecs
import copy
import sys
import warnings
def crlf(self):
    """This advances the cursor with CRLF properties.
        The cursor will line wrap and the screen may scroll.
        """
    self.cr()
    self.lf()