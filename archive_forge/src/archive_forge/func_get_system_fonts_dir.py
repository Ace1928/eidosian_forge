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
def get_system_fonts_dir():
    """Return the directories used by the system for fonts.
        """
    if LabelBase._fonts_dirs:
        return LabelBase._fonts_dirs
    fdirs = []
    if platform == 'linux':
        fdirs = ['/usr/share/fonts', '/usr/local/share/fonts', os.path.expanduser('~/.fonts'), os.path.expanduser('~/.local/share/fonts')]
    elif platform == 'macosx':
        fdirs = ['/Library/Fonts', '/System/Library/Fonts', os.path.expanduser('~/Library/Fonts')]
    elif platform == 'win':
        fdirs = [os.path.join(os.environ['SYSTEMROOT'], 'Fonts')]
    elif platform == 'ios':
        fdirs = ['/System/Library/Fonts']
    elif platform == 'android':
        fdirs = ['/system/fonts']
    else:
        raise Exception('Unknown platform: {}'.format(platform))
    fdirs.append(os.path.join(kivy_data_dir, 'fonts'))
    rdirs = []
    _font_dir_files = []
    for fdir in fdirs:
        for _dir, dirs, files in os.walk(fdir):
            _font_dir_files.extend(files)
            resource_add_path(_dir)
            rdirs.append(_dir)
    LabelBase._fonts_dirs = rdirs
    LabelBase._font_dirs_files = _font_dir_files
    return rdirs