import sys
import os
import re
import warnings
import types
import unicodedata
def serial_escape(value):
    """Escape string values that are elements of a list, for serialization."""
    return value.replace('\\', '\\\\').replace(' ', '\\ ')