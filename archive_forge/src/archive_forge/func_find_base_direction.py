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
@staticmethod
def find_base_direction(text):
    """Searches a string the first character that has a strong direction,
        according to the Unicode bidirectional algorithm. Returns `None` if
        the base direction cannot be determined, or one of `'ltr'` or `'rtl'`.

        .. versionadded: 1.10.1

        .. note:: This feature requires the Pango text provider.
        """
    return 'ltr'