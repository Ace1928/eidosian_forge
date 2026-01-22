import gettext
import os
import re
import textwrap
import warnings
from . import declarative
def base64encode(self, value):
    """
        Encode a string in base64, stripping whitespace and removing
        newlines.
        """
    return value.encode('base64').strip().replace('\n', '')