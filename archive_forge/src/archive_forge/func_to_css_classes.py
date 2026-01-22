import argparse
import io
import re
import sys
from collections import OrderedDict
from typing import Iterator, List, Optional, Set, Tuple, Union
from ansi2html.style import (
def to_css_classes(self) -> List[str]:
    css_classes: List[str] = []

    def append_unless_default(output: List[str], value: int, default: int) -> None:
        if value != default:
            css_class = 'ansi%d' % value
            output.append(css_class)

    def append_color_unless_default(output: List[str], color: Tuple[int, Optional[str]], default: int, negative: bool, neg_css_class: str) -> None:
        value, parameter = color
        if value != default:
            prefix = 'inv' if negative else 'ansi'
            css_class_index = str(value) if parameter is None else '%d-%s' % (value, parameter)
            output.append(prefix + css_class_index)
        elif negative:
            output.append(neg_css_class)
    append_unless_default(css_classes, self.intensity, ANSI_INTENSITY_NORMAL)
    append_unless_default(css_classes, self.style, ANSI_STYLE_NORMAL)
    append_unless_default(css_classes, self.blink, ANSI_BLINK_OFF)
    append_unless_default(css_classes, self.underline, ANSI_UNDERLINE_OFF)
    append_unless_default(css_classes, self.crossedout, ANSI_CROSSED_OUT_OFF)
    append_unless_default(css_classes, self.visibility, ANSI_VISIBILITY_ON)
    flip_fore_and_background = self.negative == ANSI_NEGATIVE_ON
    append_color_unless_default(css_classes, self.foreground, ANSI_FOREGROUND_DEFAULT, flip_fore_and_background, 'inv_background')
    append_color_unless_default(css_classes, self.background, ANSI_BACKGROUND_DEFAULT, flip_fore_and_background, 'inv_foreground')
    return css_classes