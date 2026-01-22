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
@property
def texture_1px(self):
    if LabelBase._texture_1px is None:
        tex = Texture.create(size=(1, 1), colorfmt='rgba')
        tex.blit_buffer(b'\x00\x00\x00\x00', colorfmt='rgba')
        LabelBase._texture_1px = tex
    return LabelBase._texture_1px