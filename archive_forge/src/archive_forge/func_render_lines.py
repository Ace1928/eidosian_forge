import re
import os
from ast import literal_eval
from functools import partial
from copy import copy
from kivy import kivy_data_dir
from kivy.config import Config
from kivy.utils import platform
from kivy.graphics.texture import Texture
from kivy.core import core_select_lib
from kivy.core.text.text_layout import layout_text, LayoutWord
from kivy.resources import resource_find, resource_add_path
from kivy.compat import PY2
from kivy.setupconfig import USE_SDL2, USE_PANGOFT2
from kivy.logger import Logger
def render_lines(self, lines, options, render_text, y, size):
    get_extents = self.get_cached_extents()
    uw, uh = options['text_size']
    padding_left = options['padding'][0]
    padding_right = options['padding'][2]
    if uw is not None:
        uww = uw - padding_left - padding_right
    w = size[0]
    sw = options['space_width']
    halign = options['halign']
    split = re.split
    find_base_dir = self.find_base_direction
    cur_base_dir = options['base_direction']
    for layout_line in lines:
        lw, lh = (layout_line.w, layout_line.h)
        line = ''
        assert len(layout_line.words) < 2
        if len(layout_line.words):
            last_word = layout_line.words[0]
            line = last_word.text
            if not cur_base_dir:
                cur_base_dir = find_base_dir(line)
        x = padding_left
        if halign == 'auto':
            if cur_base_dir and 'rtl' in cur_base_dir:
                x = max(0, int(w - lw - padding_right))
        elif halign == 'center':
            x = min(int(w - lw), max(int(padding_left), int((w - lw + padding_left - padding_right) / 2.0)))
        elif halign == 'right':
            x = max(0, int(w - lw - padding_right))
        if uw is not None and halign == 'justify' and line and (not layout_line.is_last_line):
            n, rem = divmod(max(uww - lw, 0), sw)
            n = int(n)
            words = None
            if n or rem:
                words = split(whitespace_pat, line)
            if words is not None and len(words) > 1:
                space = type(line)(' ')
                for i in range(n):
                    idx = (2 * i + 1) % (len(words) - 1)
                    words[idx] = words[idx] + space
                if rem:
                    ext = get_extents(words[-1])
                    word = LayoutWord(last_word.options, ext[0], ext[1], words[-1])
                    layout_line.words.append(word)
                    last_word.lw = uww - ext[0]
                    render_text(words[-1], x + last_word.lw, y)
                    last_word.text = line = ''.join(words[:-2])
                else:
                    last_word.lw = uww
                    last_word.text = line = ''.join(words)
                layout_line.w = uww
        if len(line):
            layout_line.x = x
            layout_line.y = y
            render_text(line, x, y)
        y += lh
    return y