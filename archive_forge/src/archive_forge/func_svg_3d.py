from __future__ import annotations
import math
import re
import numpy as np
def svg_3d(chunks, size=200, sizes=None, offset=(0, 0)):
    shape = tuple(map(sum, chunks))
    sizes = sizes or draw_sizes(shape, size=size)
    x, y, z = grid_points(chunks, sizes)
    ox, oy = offset
    xy, (mnx, mxx, mny, mxy) = svg_grid(x / 1.7, y, offset=(ox + 10, oy + 0), skew=(1, 0), size=size)
    zx, (_, _, _, max_x) = svg_grid(z, x / 1.7, offset=(ox + 10, oy + 0), skew=(0, 1), size=size)
    zy, (min_z, max_z, min_y, max_y) = svg_grid(z, y, offset=(ox + max_x + 10, oy + max_x), skew=(0, 0), size=size)
    header = '<svg width="%d" height="%d" style="stroke:rgb(0,0,0);stroke-width:1" >\n' % (max_z + 50, max_y + 50)
    footer = '\n</svg>'
    if shape[1] >= 100:
        rotate = -90
    else:
        rotate = 0
    text = ['', '  <!-- Text -->', '  <text x="%f" y="%f" %s >%d</text>' % ((min_z + max_z) / 2, max_y + 20, text_style, shape[2]), '  <text x="%f" y="%f" %s transform="rotate(%d,%f,%f)">%d</text>' % (max_z + 20, (min_y + max_y) / 2, text_style, rotate, max_z + 20, (min_y + max_y) / 2, shape[1]), '  <text x="%f" y="%f" %s transform="rotate(45,%f,%f)">%d</text>' % ((mnx + mxx) / 2 - 10, mxy - (mxx - mnx) / 2 + 20, text_style, (mnx + mxx) / 2 - 10, mxy - (mxx - mnx) / 2 + 20, shape[0])]
    return header + '\n'.join(xy + zx + zy + text) + footer