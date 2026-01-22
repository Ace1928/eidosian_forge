from __future__ import annotations
import re
from functools import lru_cache
from . import Image
@lru_cache
def getrgb(color):
    """
     Convert a color string to an RGB or RGBA tuple. If the string cannot be
     parsed, this function raises a :py:exc:`ValueError` exception.

    .. versionadded:: 1.1.4

    :param color: A color string
    :return: ``(red, green, blue[, alpha])``
    """
    if len(color) > 100:
        msg = 'color specifier is too long'
        raise ValueError(msg)
    color = color.lower()
    rgb = colormap.get(color, None)
    if rgb:
        if isinstance(rgb, tuple):
            return rgb
        colormap[color] = rgb = getrgb(rgb)
        return rgb
    if re.match('#[a-f0-9]{3}$', color):
        return (int(color[1] * 2, 16), int(color[2] * 2, 16), int(color[3] * 2, 16))
    if re.match('#[a-f0-9]{4}$', color):
        return (int(color[1] * 2, 16), int(color[2] * 2, 16), int(color[3] * 2, 16), int(color[4] * 2, 16))
    if re.match('#[a-f0-9]{6}$', color):
        return (int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16))
    if re.match('#[a-f0-9]{8}$', color):
        return (int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16), int(color[7:9], 16))
    m = re.match('rgb\\(\\s*(\\d+)\\s*,\\s*(\\d+)\\s*,\\s*(\\d+)\\s*\\)$', color)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.match('rgb\\(\\s*(\\d+)%\\s*,\\s*(\\d+)%\\s*,\\s*(\\d+)%\\s*\\)$', color)
    if m:
        return (int(int(m.group(1)) * 255 / 100.0 + 0.5), int(int(m.group(2)) * 255 / 100.0 + 0.5), int(int(m.group(3)) * 255 / 100.0 + 0.5))
    m = re.match('hsl\\(\\s*(\\d+\\.?\\d*)\\s*,\\s*(\\d+\\.?\\d*)%\\s*,\\s*(\\d+\\.?\\d*)%\\s*\\)$', color)
    if m:
        from colorsys import hls_to_rgb
        rgb = hls_to_rgb(float(m.group(1)) / 360.0, float(m.group(3)) / 100.0, float(m.group(2)) / 100.0)
        return (int(rgb[0] * 255 + 0.5), int(rgb[1] * 255 + 0.5), int(rgb[2] * 255 + 0.5))
    m = re.match('hs[bv]\\(\\s*(\\d+\\.?\\d*)\\s*,\\s*(\\d+\\.?\\d*)%\\s*,\\s*(\\d+\\.?\\d*)%\\s*\\)$', color)
    if m:
        from colorsys import hsv_to_rgb
        rgb = hsv_to_rgb(float(m.group(1)) / 360.0, float(m.group(2)) / 100.0, float(m.group(3)) / 100.0)
        return (int(rgb[0] * 255 + 0.5), int(rgb[1] * 255 + 0.5), int(rgb[2] * 255 + 0.5))
    m = re.match('rgba\\(\\s*(\\d+)\\s*,\\s*(\\d+)\\s*,\\s*(\\d+)\\s*,\\s*(\\d+)\\s*\\)$', color)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))
    msg = f'unknown color specifier: {repr(color)}'
    raise ValueError(msg)