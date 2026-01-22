from __future__ import annotations
import math
import numbers
import struct
from . import Image, ImageColor
def multiline_textbbox(self, xy, text, font=None, anchor=None, spacing=4, align='left', direction=None, features=None, language=None, stroke_width=0, embedded_color=False, *, font_size=None):
    if direction == 'ttb':
        msg = 'ttb direction is unsupported for multiline text'
        raise ValueError(msg)
    if anchor is None:
        anchor = 'la'
    elif len(anchor) != 2:
        msg = 'anchor must be a 2 character string'
        raise ValueError(msg)
    elif anchor[1] in 'tb':
        msg = 'anchor not supported for multiline text'
        raise ValueError(msg)
    if font is None:
        font = self._getfont(font_size)
    widths = []
    max_width = 0
    lines = self._multiline_split(text)
    line_spacing = self._multiline_spacing(font, spacing, stroke_width)
    for line in lines:
        line_width = self.textlength(line, font, direction=direction, features=features, language=language, embedded_color=embedded_color)
        widths.append(line_width)
        max_width = max(max_width, line_width)
    top = xy[1]
    if anchor[1] == 'm':
        top -= (len(lines) - 1) * line_spacing / 2.0
    elif anchor[1] == 'd':
        top -= (len(lines) - 1) * line_spacing
    bbox = None
    for idx, line in enumerate(lines):
        left = xy[0]
        width_difference = max_width - widths[idx]
        if anchor[0] == 'm':
            left -= width_difference / 2.0
        elif anchor[0] == 'r':
            left -= width_difference
        if align == 'left':
            pass
        elif align == 'center':
            left += width_difference / 2.0
        elif align == 'right':
            left += width_difference
        else:
            msg = 'align must be "left", "center" or "right"'
            raise ValueError(msg)
        bbox_line = self.textbbox((left, top), line, font, anchor, direction=direction, features=features, language=language, stroke_width=stroke_width, embedded_color=embedded_color)
        if bbox is None:
            bbox = bbox_line
        else:
            bbox = (min(bbox[0], bbox_line[0]), min(bbox[1], bbox_line[1]), max(bbox[2], bbox_line[2]), max(bbox[3], bbox_line[3]))
        top += line_spacing
    if bbox is None:
        return (xy[0], xy[1], xy[0], xy[1])
    return bbox