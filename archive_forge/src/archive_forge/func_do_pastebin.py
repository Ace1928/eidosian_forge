import abc
import code
import inspect
import os
import pkgutil
import pydoc
import shlex
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
from abc import abstractmethod
from dataclasses import dataclass
from itertools import takewhile
from pathlib import Path
from types import ModuleType, TracebackType
from typing import (
from ._typing_compat import Literal
from pygments.lexers import Python3Lexer
from pygments.token import Token, _TokenType
from . import autocomplete, inspection, simpleeval
from .config import getpreferredencoding, Config
from .formatter import Parenthesis
from .history import History
from .lazyre import LazyReCompile
from .paste import PasteHelper, PastePinnwand, PasteFailed
from .patch_linecache import filename_for_console_input
from .translations import _, ngettext
from .importcompletion import ModuleGatherer
def do_pastebin(self, s) -> Optional[str]:
    """Actually perform the upload."""
    paste_url: str
    if s == self.prev_pastebin_content:
        self.interact.notify(_('Duplicate pastebin. Previous URL: %s. Removal URL: %s') % (self.prev_pastebin_url, self.prev_removal_url), 10)
        return self.prev_pastebin_url
    self.interact.notify(_('Posting data to pastebin...'))
    try:
        paste_url, removal_url = self.paster.paste(s)
    except PasteFailed as e:
        self.interact.notify(_('Upload failed: %s') % e)
        return None
    self.prev_pastebin_content = s
    self.prev_pastebin_url = paste_url
    self.prev_removal_url = removal_url if removal_url is not None else ''
    if removal_url is not None:
        self.interact.notify(_('Pastebin URL: %s - Removal URL: %s') % (paste_url, removal_url), 10)
    else:
        self.interact.notify(_('Pastebin URL: %s') % (paste_url,), 10)
    return paste_url