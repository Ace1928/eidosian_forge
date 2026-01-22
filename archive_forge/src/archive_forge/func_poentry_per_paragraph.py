import inspect
import os
import sys
import breezy
from . import commands as _mod_commands
from . import errors, help_topics, option
from . import plugin as _mod_plugin
from .i18n import gettext
from .trace import mutter, note
def poentry_per_paragraph(self, path, lineno, msgid, include=None):
    paragraphs = msgid.split('\n\n')
    if include is not None:
        paragraphs = filter(include, paragraphs)
    for p in paragraphs:
        self.poentry(path, lineno, p)
        lineno += p.count('\n') + 2