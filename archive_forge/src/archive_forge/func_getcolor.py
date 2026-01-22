from __future__ import annotations
import array
from . import GimpGradientFile, GimpPaletteFile, ImageColor, PaletteFile
def getcolor(self, color, image=None):
    """Given an rgb tuple, allocate palette entry.

        .. warning:: This method is experimental.
        """
    if self.rawmode:
        msg = 'palette contains raw palette data'
        raise ValueError(msg)
    if isinstance(color, tuple):
        if self.mode == 'RGB':
            if len(color) == 4:
                if color[3] != 255:
                    msg = 'cannot add non-opaque RGBA color to RGB palette'
                    raise ValueError(msg)
                color = color[:3]
        elif self.mode == 'RGBA':
            if len(color) == 3:
                color += (255,)
        try:
            return self.colors[color]
        except KeyError as e:
            index = self._new_color_index(image, e)
            self.colors[color] = index
            if index * 3 < len(self.palette):
                self._palette = self.palette[:index * 3] + bytes(color) + self.palette[index * 3 + 3:]
            else:
                self._palette += bytes(color)
            self.dirty = 1
            return index
    else:
        msg = f'unknown color specifier: {repr(color)}'
        raise ValueError(msg)