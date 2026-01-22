import sys
import os
import re
import warnings
import types
import unicodedata
def note_substitution_ref(self, subref, refname):
    subref['refname'] = whitespace_normalize_name(refname)