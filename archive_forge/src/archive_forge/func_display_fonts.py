import sys
import os
import pygame as pg
def display_fonts(self):
    """
        Display the visible fonts based on the y_offset value(updated in
        handle_events) and the height of the pygame window.
        """
    pg.display.set_caption('Font Viewer')
    display = pg.display.get_surface()
    clock = pg.time.Clock()
    center = display.get_width() // 2
    while True:
        display.fill(self.back_color)
        for surface, top in self.font_surfaces:
            bottom = top + surface.get_height()
            if bottom >= self.y_offset and top <= self.y_offset + display.get_height():
                x = center - surface.get_width() // 2
                display.blit(surface, (x, top - self.y_offset))
        if not self.handle_events():
            break
        pg.display.flip()
        clock.tick(30)