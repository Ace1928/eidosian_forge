import io
import math
import os
import typing
import weakref
def create_label_str(label):
    """Convert Python label dict to correspnding PDF rule string.

        Args:
            label: (dict) build rule for the label.
        Returns:
            PDF label rule string wrapped in "<<", ">>".
        """
    s = '%i<<' % label['startpage']
    if label.get('prefix', '') != '':
        s += '/P(%s)' % label['prefix']
    if label.get('style', '') != '':
        s += '/S/%s' % label['style']
    if label.get('firstpagenum', 1) > 1:
        s += '/St %i' % label['firstpagenum']
    s += '>>'
    return s