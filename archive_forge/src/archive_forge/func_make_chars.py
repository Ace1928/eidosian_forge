import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
def make_chars(page, clip=None):
    """Extract text as "rawdict" to fill CHARS."""
    global CHARS, TEXTPAGE
    page_number = page.number + 1
    page_height = page.rect.height
    ctm = page.transformation_matrix
    TEXTPAGE = page.get_textpage(clip=clip, flags=TEXTFLAGS_TEXT)
    blocks = page.get_text('rawdict', textpage=TEXTPAGE)['blocks']
    doctop_base = page_height * page.number
    for block in blocks:
        for line in block['lines']:
            ldir = line['dir']
            matrix = Matrix(ldir[0], -ldir[1], ldir[1], ldir[0], 0, 0)
            if ldir[1] == 0:
                upright = True
            else:
                upright = False
            for span in sorted(line['spans'], key=lambda s: s['bbox'][0]):
                fontname = span['font']
                fontsize = span['size']
                color = sRGB_to_pdf(span['color'])
                for char in sorted(span['chars'], key=lambda c: c['bbox'][0]):
                    bbox = Rect(char['bbox'])
                    bbox_ctm = bbox * ctm
                    origin = Point(char['origin']) * ctm
                    matrix.e = origin.x
                    matrix.f = origin.y
                    text = char['c']
                    char_dict = {'adv': bbox.x1 - bbox.x0 if upright else bbox.y1 - bbox.y0, 'bottom': bbox.y1, 'doctop': bbox.y0 + doctop_base, 'fontname': fontname, 'height': bbox.y1 - bbox.y0, 'matrix': tuple(matrix), 'ncs': 'DeviceRGB', 'non_stroking_color': color, 'non_stroking_pattern': None, 'object_type': 'char', 'page_number': page_number, 'size': fontsize if upright else bbox.y1 - bbox.y0, 'stroking_color': color, 'stroking_pattern': None, 'text': text, 'top': bbox.y0, 'upright': upright, 'width': bbox.x1 - bbox.x0, 'x0': bbox.x0, 'x1': bbox.x1, 'y0': bbox_ctm.y0, 'y1': bbox_ctm.y1}
                    CHARS.append(char_dict)