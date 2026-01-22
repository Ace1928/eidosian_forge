import os
import sys
from pygments.formatter import Formatter
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
import subprocess
def get_text_size(self, text):
    """
        Get the text size (width, height).
        """
    font = self.fonts['NORMAL']
    if hasattr(font, 'getbbox'):
        return font.getbbox(text)[2:4]
    else:
        return font.getsize(text)