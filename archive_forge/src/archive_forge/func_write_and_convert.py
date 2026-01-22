import re
import sys
import os
from .ansi import AnsiFore, AnsiBack, AnsiStyle, Style, BEL
from .winterm import enable_vt_processing, WinTerm, WinColor, WinStyle
from .win32 import windll, winapi_test
def write_and_convert(self, text):
    """
        Write the given text to our wrapped stream, stripping any ANSI
        sequences from the text, and optionally converting them into win32
        calls.
        """
    cursor = 0
    text = self.convert_osc(text)
    for match in self.ANSI_CSI_RE.finditer(text):
        start, end = match.span()
        self.write_plain_text(text, cursor, start)
        self.convert_ansi(*match.groups())
        cursor = end
    self.write_plain_text(text, cursor, len(text))