import sys
import os
import re
import warnings
import types
import unicodedata
def note_symbol_footnote_ref(self, ref):
    self.set_id(ref)
    self.symbol_footnote_refs.append(ref)