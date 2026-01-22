import sys
import os
import re
import warnings
import types
import unicodedata
def fully_normalize_name(name):
    """Return a case- and whitespace-normalized name."""
    return ' '.join(name.lower().split())