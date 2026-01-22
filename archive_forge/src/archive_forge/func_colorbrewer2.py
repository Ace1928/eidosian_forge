from __future__ import absolute_import
from __future__ import print_function
import webbrowser
from ..palette import Palette
from .colorbrewer_all_schemes import COLOR_MAPS
def colorbrewer2(self):
    """
        View this color map at colorbrewer2.org. This will open
        colorbrewer2.org in your default web browser.

        """
    webbrowser.open_new_tab(self.colorbrewer2_url)