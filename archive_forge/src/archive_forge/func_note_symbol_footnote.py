import sys
import os
import re
import warnings
import types
import unicodedata
def note_symbol_footnote(self, footnote):
    self.set_id(footnote)
    self.symbol_footnotes.append(footnote)