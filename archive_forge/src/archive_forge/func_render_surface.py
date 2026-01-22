import sys
import os
import pygame as pg
def render_surface(self):
    """
        Note: this method uses twice the memory and is only called if
        big_surface is set to true or big is added to the command line.

        Optionally generates one large buffer to draw all the font surfaces
        into. This is necessary to save the display to a png file and may
        be useful for testing large surfaces.
        """
    large_surface = pg.surface.Surface((self.max_width, self.total_height)).convert()
    large_surface.fill(self.back_color)
    print('scrolling surface created')
    byte_size = large_surface.get_bytesize()
    total_size = byte_size * (self.max_width * self.total_height)
    print('Surface Size = {}x{} @ {}bpp: {:,.3f}mb'.format(self.max_width, self.total_height, byte_size, total_size / 1000000.0))
    y = 0
    center = int(self.max_width / 2)
    for surface, top in self.font_surfaces:
        w = surface.get_width()
        x = center - int(w / 2)
        large_surface.blit(surface, (x, y))
        y += surface.get_height()
    self.max_y = large_surface.get_height() - pg.display.get_surface().get_height()
    self.surface = large_surface