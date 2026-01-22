import weakref
from inspect import isroutine
from copy import copy
from time import time
from kivy.eventmanager import MODE_DEFAULT_DISPATCH
from kivy.vector import Vector
def scale_for_screen(self, w, h, p=None, rotation=0, smode='None', kheight=0):
    """Scale position for the screen.

        .. versionchanged:: 2.1.0
            Max value for `x`, `y` and `z` is changed respectively to `w` - 1,
            `h` - 1 and `p` - 1.
        """
    x_max, y_max = (max(0, w - 1), max(0, h - 1))
    absolute = self.to_absolute_pos
    self.x, self.y = absolute(self.sx, self.sy, x_max, y_max, rotation)
    self.ox, self.oy = absolute(self.osx, self.osy, x_max, y_max, rotation)
    self.px, self.py = absolute(self.psx, self.psy, x_max, y_max, rotation)
    z_max = 0 if p is None else max(0, p - 1)
    self.z = self.sz * z_max
    self.oz = self.osz * z_max
    self.pz = self.psz * z_max
    if smode:
        if smode == 'pan' or smode == 'below_target':
            self.y -= kheight
            self.oy -= kheight
            self.py -= kheight
        elif smode == 'scale':
            offset = kheight * (self.y - h) / (h - kheight)
            self.y += offset
            self.oy += offset
            self.py += offset
    self.dx = self.x - self.px
    self.dy = self.y - self.py
    self.dz = self.z - self.pz
    self.pos = (self.x, self.y)