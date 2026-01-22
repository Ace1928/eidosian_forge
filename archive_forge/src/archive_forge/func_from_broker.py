import enchant
import gettext
import logging
import re
import sys
from collections import UserList
from ._pylocales import code_to_name as _code_to_name
from ._pylocales import LanguageNotFound, CountryNotFound
from gi.repository import Gio, GLib, GObject
@classmethod
def from_broker(cls, broker):
    return cls(sorted([(language, code_to_name(language)) for language in broker.list_languages()], key=lambda language: language[1]))