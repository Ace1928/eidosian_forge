import sys
import os
import re
import warnings
import types
import unicodedata
def get_language_code(self, fallback=''):
    """Return node's language tag.

        Look iteratively in self and parents for a class argument
        starting with ``language-`` and return the remainder of it
        (which should be a `BCP49` language tag) or the `fallback`.
        """
    for cls in self.get('classes', []):
        if cls.startswith('language-'):
            return cls[9:]
    try:
        return self.parent.get_language(fallback)
    except AttributeError:
        return fallback