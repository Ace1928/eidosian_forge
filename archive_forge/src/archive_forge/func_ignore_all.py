import enchant
import gettext
import logging
import re
import sys
from collections import UserList
from ._pylocales import code_to_name as _code_to_name
from ._pylocales import LanguageNotFound, CountryNotFound
from gi.repository import Gio, GLib, GObject
def ignore_all(self, word):
    """
        Ignores a word for the current session.

        :param word: The word to ignore.
        """
    self._dictionary.add_to_session(word)
    self.recheck()