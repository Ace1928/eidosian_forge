import os
from copy import deepcopy, copy
from reportlab.lib.colors import gray, lightgrey
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.lib.styles import _baseFontName
from reportlab.lib.utils import strTypes, rl_safe_exec, annotateException
from reportlab.lib.abag import ABag
from reportlab.pdfbase import pdfutils
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.rl_config import _FUZZ, overlapAttachedSpace, ignoreContainerActions, listWrapOnFakeWidth
from reportlab.lib.sequencer import _type2formatter
from reportlab.lib.styles import ListStyle
def splitLines(lines, maximum_length, split_characters, new_line_characters):
    if split_characters is None:
        split_characters = SPLIT_CHARS
    if new_line_characters is None:
        new_line_characters = ''
    lines_splitted = []
    for line in lines:
        if len(line) > maximum_length:
            splitLine(line, lines_splitted, maximum_length, split_characters, new_line_characters)
        else:
            lines_splitted.append(line)
    return lines_splitted