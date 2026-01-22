import unicodedata
from pyglet.gl import *
from pyglet import image
def set_bearings(self, baseline, left_side_bearing, advance, x_offset=0, y_offset=0):
    """Set metrics for this glyph.

        :Parameters:
            `baseline` : int
                Distance from the bottom of the glyph to its baseline;
                typically negative.
            `left_side_bearing` : int
                Distance to add to the left edge of the glyph.
            `advance` : int
                Distance to move the horizontal advance to the next glyph.
            `offset_x` : int
                Distance to move the glyph horizontally from its default position.
            `offset_y` : int
                Distance to move the glyph vertically from its default position.
        """
    self.baseline = baseline
    self.lsb = left_side_bearing
    self.advance = advance
    self.vertices = (left_side_bearing + x_offset, -baseline + y_offset, left_side_bearing + self.width + x_offset, -baseline + self.height + y_offset)