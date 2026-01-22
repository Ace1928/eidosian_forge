import enchant
import gettext
import logging
import re
import sys
from collections import UserList
from ._pylocales import code_to_name as _code_to_name
from ._pylocales import LanguageNotFound, CountryNotFound
from gi.repository import Gio, GLib, GObject
def move_through_extra_chars():
    tmp = loc.copy()
    tmp.backward_char()
    moved = False
    while self.is_extra_word_char(tmp):
        moved = True
        loc.assign(tmp)
        if not tmp.backward_char():
            break
    return moved