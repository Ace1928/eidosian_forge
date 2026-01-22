from __future__ import absolute_import, unicode_literals
import pyperclip
from prompt_toolkit.selection import SelectionType
from .base import Clipboard, ClipboardData

    Clipboard that synchronizes with the Windows/Mac/Linux system clipboard,
    using the pyperclip module.
    