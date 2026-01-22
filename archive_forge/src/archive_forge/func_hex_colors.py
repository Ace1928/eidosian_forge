from __future__ import absolute_import
import sys
@property
def hex_colors(self):
    """
        Colors as a tuple of hex strings. (e.g. '#A912F4')

        """
    hc = []
    for color in self.colors:
        h = '#' + ''.join(('{0:>02}'.format(hex(c)[2:].upper()) for c in color))
        hc.append(h)
    return hc