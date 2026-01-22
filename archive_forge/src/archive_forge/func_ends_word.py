import enchant
import gettext
import logging
import re
import sys
from collections import UserList
from ._pylocales import code_to_name as _code_to_name
from ._pylocales import LanguageNotFound, CountryNotFound
from gi.repository import Gio, GLib, GObject
def ends_word(self, loc):
    if loc.ends_word():
        if loc.is_end():
            return True
        else:
            tmp = loc.copy()
            tmp.forward_char()
            return not self.is_extra_word_char(tmp)
    else:
        return False